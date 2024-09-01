import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import joblib
import calendar
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import requests
import tropycal.tracks as tracks
import sys
import pickle
from scipy.fft import fft, ifft
from scipy.interpolate import interp1d
import glob

# Constants
CHECKPOINT_DIR = 'checkpoints'
CHECKPOINT_INTERVAL = 10
FOURIER_DATA_DIR = 'fourier_data'
IBTRACS_URL = 'https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.WP.list.v04r00.csv'
IBTRACS_LOCAL_PATH = 'ibtracs_wp.csv'
ONI_DATA_PATH = os.path.join(os.getcwd(), 'oni_data.csv')
MODEL_PATH = os.path.join(os.getcwd(), 'hurricane_rnn_model.pth')
SCALER_PATH = os.path.join(os.getcwd(), 'scaler.joblib')
FOURIER_CACHE_FILE = 'typhoon_routes_fourier.pkl'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_ibtracs_data():
    cache_file = 'ibtracs.cache'
    if os.path.exists(cache_file):
        print("Checking for IBTrACS data updates...")
        try:
            response = requests.head(IBTRACS_URL)
            remote_last_modified = datetime.strptime(response.headers['Last-Modified'], '%a, %d %b %Y %H:%M:%S GMT')
            local_last_modified = datetime.fromtimestamp(os.path.getmtime(cache_file))

            if remote_last_modified <= local_last_modified:
                print("Local IBTrACS cache is up to date. Loading from cache...")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            
            print("Remote data is newer. Updating IBTrACS data...")
            response = requests.get(IBTRACS_URL)
            response.raise_for_status()
            
            with open(IBTRACS_LOCAL_PATH, 'w') as f:
                f.write(response.text)
            
            ibtracs = tracks.TrackDataset(basin='west_pacific', source='ibtracs', ibtracs_url=IBTRACS_LOCAL_PATH)
            
            with open(cache_file, 'wb') as f:
                pickle.dump(ibtracs, f)
            
            print("IBTrACS data updated and cache refreshed.")
            return ibtracs

        except requests.RequestException as e:
            print(f"Error checking or downloading data: {e}")
            print("Using existing local cache.")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

    else:
        print("No local cache found. Downloading IBTrACS data...")
        try:
            response = requests.get(IBTRACS_URL)
            response.raise_for_status()
            
            with open(IBTRACS_LOCAL_PATH, 'w') as f:
                f.write(response.text)
            
            ibtracs = tracks.TrackDataset(basin='west_pacific', source='ibtracs', ibtracs_url=IBTRACS_LOCAL_PATH)
            
            with open(cache_file, 'wb') as f:
                pickle.dump(ibtracs, f)
            
            print("IBTrACS data downloaded and cached.")
            return ibtracs

        except requests.RequestException as e:
            print(f"Error downloading data: {e}")
            print("Unable to load IBTrACS data.")
            return None

ibtracs = load_ibtracs_data()

def filter_west_pacific_coordinates(lons, lats):
    mask = (100 <= lons) & (lons <= 180) & (0 <= lats) & (lats <= 40)
    return lons[mask], lats[mask]

def compute_fourier_coefficients(lons, lats, n_components=10):
    t = np.linspace(0, 2*np.pi, len(lons))
    coeffs_lon = fft(lons)[:n_components+1]
    coeffs_lat = fft(lats)[:n_components+1]
    return np.concatenate([coeffs_lon.real, coeffs_lon.imag, coeffs_lat.real, coeffs_lat.imag])

def inverse_fourier_transform(coeffs, n_points=200):
    n_components = len(coeffs) // 4
    coeffs_lon = coeffs[:n_components] + 1j * coeffs[n_components:2*n_components]
    coeffs_lat = coeffs[2*n_components:3*n_components] + 1j * coeffs[3*n_components:]
    
    t = np.linspace(0, 2*np.pi, n_points)
    lons = np.real(np.fft.ifft(coeffs_lon, n=n_points))
    lats = np.real(np.fft.ifft(coeffs_lat, n=n_points))
    
    # Apply some randomness to make the path less straight
    lons += np.random.normal(0, 0.5, n_points)
    lats += np.random.normal(0, 0.5, n_points)
    
    # Smooth the path
    lons = np.convolve(lons, np.ones(5)/5, mode='same')
    lats = np.convolve(lats, np.ones(5)/5, mode='same')
    
    return lons, lats

def prepare_data(start_year, end_year, sequence_length, n_fourier_components=10):
    if os.path.exists(FOURIER_CACHE_FILE):
        with open(FOURIER_CACHE_FILE, 'rb') as f:
            cache = pickle.load(f)
        if cache['start_year'] == start_year and cache['end_year'] == end_year:
            print("Loading existing Fourier data...")
            data = cache['data']
            
            if 'max_coeff_length' not in cache:
                max_coeff_length = max(len(d['coeffs']) for d in data)
                
                for item in data:
                    item['coeffs'] = np.pad(item['coeffs'], (0, max_coeff_length - len(item['coeffs'])))
                
                cache['max_coeff_length'] = max_coeff_length
                with open(FOURIER_CACHE_FILE, 'wb') as f:
                    pickle.dump(cache, f)
            else:
                max_coeff_length = cache['max_coeff_length']
            
            return data, max_coeff_length
    
    print("Preparing typhoon data...")
    data = []
    max_coeff_length = 0
    
    for year in range(start_year, end_year + 1):
        season = ibtracs.get_season(year)
        for storm_id in season.summary()['id']:
            storm = ibtracs.get_storm(storm_id)
            lons, lats = filter_west_pacific_coordinates(np.array(storm.lon), np.array(storm.lat))
            
            if len(lons) < max(sequence_length + 1, n_fourier_components * 2):
                continue
            
            if isinstance(storm.time[0], datetime):
                start_date = storm.time[0]
                end_date = storm.time[-1]
            else:
                start_date = datetime.strptime(storm.time[0], '%Y-%m-%d %H:%M:%S')
                end_date = datetime.strptime(storm.time[-1], '%Y-%m-%d %H:%M:%S')
            
            lifetime = (end_date - start_date).days + 1
            
            fourier_coeffs = compute_fourier_coefficients(lons, lats, n_fourier_components)
            max_coeff_length = max(max_coeff_length, len(fourier_coeffs))
            
            data.append({
                'coeffs': fourier_coeffs,
                'start_point': np.array([lons[0], lats[0]]),
                'end_point': np.array([lons[-1], lats[-1]]),
                'lifetime': lifetime,
                'month': start_date.month,
                'oni': get_oni_value(start_date)
            })
    
    for item in data:
        item['coeffs'] = np.pad(item['coeffs'], (0, max_coeff_length - len(item['coeffs'])))
    
    cache = {
        'start_year': start_year,
        'end_year': end_year,
        'data': data,
        'max_coeff_length': max_coeff_length
    }
    with open(FOURIER_CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)
    
    return data, max_coeff_length

def get_oni_value(date):
    # This is a placeholder implementation. Replace with actual ONI data lookup.
    # For now, we'll return a random value between -2 and 2
    return np.random.uniform(-2, 2)

class HurricaneRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2):
        super(HurricaneRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size + 3, 1)  # +3 for lifetime and spawn_location

    def forward(self, x, lifetime, spawn_location):
        #print(f"HurricaneRNN input shapes: x={x.shape}, lifetime={lifetime.shape}, spawn_location={spawn_location.shape}")
        
        batch_size = x.size(0)
        
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        
        lifetime = lifetime.view(-1, 1)
        spawn_location = spawn_location.view(-1, 2)
        
        out = torch.cat([out, lifetime, spawn_location], dim=1)
        out = self.fc(out)
        
        #print(f"HurricaneRNN output shape: {out.shape}")
        return out  # Remove the squeeze here

class MultiCoefficientPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_coefficients, num_layers=2, dropout=0.2):
        super(MultiCoefficientPredictor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_coefficients = num_coefficients
        self.models = nn.ModuleList([HurricaneRNN(input_size, hidden_size, num_layers, dropout) for _ in range(num_coefficients)])

    def forward(self, x, lifetime, spawn_location):
        outputs = [model(x, lifetime, spawn_location) for model in self.models]
        return torch.cat(outputs, dim=1)  # Change dim to 1

    def get_input_size(self):
        return self.input_size

    def get_hidden_size(self):
        return self.hidden_size

    def get_output_size(self):
        return self.num_coefficients
 
def fourier_series_equation(coeffs):
    n_components = (len(coeffs) // 4)
    a0_lon, a0_lat = coeffs[0], coeffs[2*n_components]
    equation_lon = f"Longitude = {a0_lon:.4f}"
    equation_lat = f"Latitude = {a0_lat:.4f}"
    for i in range(1, n_components):
        ai_lon, bi_lon = coeffs[i], coeffs[n_components + i]
        ai_lat, bi_lat = coeffs[2*n_components + i], coeffs[3*n_components + i]
        equation_lon += f" + {ai_lon:.4f}*cos({i}t) + {bi_lon:.4f}*sin({i}t)"
        equation_lat += f" + {ai_lat:.4f}*cos({i}t) + {bi_lat:.4f}*sin({i}t)"
    return f"{equation_lon}\n{equation_lat}"

def save_checkpoint(model, optimizer, epoch, batch, X_scaler, y_scaler, start_scaler, lifetime_scaler, spawn_scaler, filename):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'batch': batch,
        'X_scaler': X_scaler,
        'y_scaler': y_scaler,
        'start_scaler': start_scaler,
        'lifetime_scaler': lifetime_scaler,
        'spawn_scaler': spawn_scaler,
    }, os.path.join(CHECKPOINT_DIR, filename))

def get_newest_checkpoint():
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, '*.pt'))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getctime)

def initialize_model(input_size, hidden_size, output_size, fc_start_input_size):
    model = HurricaneRNN(input_size, hidden_size, output_size, fc_start_input_size)
    return model

def train_model(data, max_coeff_length, sequence_length, nb_epochs=1600, batch_size=32, learning_rate=0.001, resume_from=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X = np.array([d['coeffs'] for d in data])
    y = np.array([d['coeffs'] for d in data])  # Use coeffs as target instead of end_point
    start_points = np.array([d['start_point'] for d in data])
    lifetimes = np.array([d['lifetime'] for d in data])
    spawn_locations = np.array([d['start_point'] for d in data])
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_start = MinMaxScaler()
    scaler_lifetime = MinMaxScaler()
    scaler_spawn = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    start_points_scaled = scaler_start.fit_transform(start_points)
    lifetimes_scaled = scaler_lifetime.fit_transform(lifetimes.reshape(-1, 1)).flatten()
    spawn_locations_scaled = scaler_spawn.fit_transform(spawn_locations)
    
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    y_tensor = torch.FloatTensor(y_scaled).to(device)
    start_points_tensor = torch.FloatTensor(start_points_scaled).to(device)
    lifetimes_tensor = torch.FloatTensor(lifetimes_scaled).to(device)
    spawn_locations_tensor = torch.FloatTensor(spawn_locations_scaled).to(device)
    
    dataset = TensorDataset(X_tensor, y_tensor, start_points_tensor, lifetimes_tensor, spawn_locations_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    input_size = X.shape[1]  # Use the number of features in X as input_size
    hidden_size = 256
    num_coefficients = y.shape[1]

    model = MultiCoefficientPredictor(input_size, hidden_size, num_coefficients).to(device)

   
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)
    
   
    start_epoch = 0
    if resume_from:
        print(f"Loading checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming from epoch {start_epoch}")
    else:
        print("Starting training from scratch.")
    
    total_epochs = start_epoch + nb_epochs
    for epoch in tqdm(range(start_epoch, total_epochs)):
        model.train()
        total_loss = 0
        for batch, (batch_X, batch_y, batch_start, batch_lifetime, batch_spawn) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(batch_X, batch_lifetime, batch_spawn)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_tensor, lifetimes_tensor, spawn_locations_tensor), y_tensor).item()
        
        scheduler.step(val_loss)
        
        print(f"Epoch [{epoch+1}/{total_epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(model, optimizer, epoch, batch, scaler_X, scaler_y, scaler_start, scaler_lifetime, scaler_spawn, f'checkpoint_epoch{epoch+1}.pt')
    
    return model, scaler_X, scaler_y, scaler_lifetime, scaler_spawn

def generate_future_route(model, scaler_X, scaler_y, month, oni, lifetime, spawn_location, n_points=200):
    try:
        device = next(model.parameters()).device
        
        input_features = np.zeros(model.get_input_size())
        input_features[0] = month
        input_features[1] = oni
        input_features[2] = lifetime
        input_features[3:5] = spawn_location
        
        scaled_input = scaler_X.transform(input_features.reshape(1, -1))
        
        X = torch.FloatTensor(scaled_input).to(device)
        lifetime_tensor = torch.tensor([[lifetime]], dtype=torch.float32).to(device)
        spawn_location_tensor = torch.tensor([spawn_location], dtype=torch.float32).to(device)
        
        model.eval()
        with torch.no_grad():
            pred_coeffs = model(X, lifetime_tensor, spawn_location_tensor)
        
        coeffs = pred_coeffs.cpu().numpy()
        coeffs = scaler_y.inverse_transform(coeffs).flatten()
        
        lons, lats = inverse_fourier_transform(coeffs, n_points=n_points)
        
        route = np.column_stack((lons, lats))
        route[:, 0] = np.clip(route[:, 0], 100, 180)
        route[:, 1] = np.clip(route[:, 1], 0, 40)
        
        route_description = describe_route(route)
        
        return route, route_description
    except Exception as e:
        print(f"Error in generate_future_route for month {month}: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Error: Could not generate route for month {month}"

def describe_route(route):
    start_point = route[0]
    end_point = route[-1]
    mid_point = route[len(route)//2]
    
    description = f"Start: ({start_point[0]:.2f}°E, {start_point[1]:.2f}°N)\n"
    description += f"Mid-point: ({mid_point[0]:.2f}°E, {mid_point[1]:.2f}°N)\n"
    description += f"End: ({end_point[0]:.2f}°E, {end_point[1]:.2f}°N)\n"
    
    # Calculate overall direction
    direction = end_point - start_point
    compass_direction = get_compass_direction(direction)
    description += f"Overall direction: {compass_direction}\n"
    
    # Calculate total distance
    total_distance = np.sum(np.sqrt(np.sum(np.diff(route, axis=0)**2, axis=1)))
    description += f"Total distance: {total_distance:.2f} degrees"
    
    return description

def get_compass_direction(direction):
    angle = np.arctan2(direction[1], direction[0]) * 180 / np.pi
    if -22.5 <= angle < 22.5:
        return "East"
    elif 22.5 <= angle < 67.5:
        return "Northeast"
    elif 67.5 <= angle < 112.5:
        return "North"
    elif 112.5 <= angle < 157.5:
        return "Northwest"
    elif angle >= 157.5 or angle < -157.5:
        return "West"
    elif -157.5 <= angle < -112.5:
        return "Southwest"
    elif -112.5 <= angle < -67.5:
        return "South"
    else:
        return "Southeast"

def calculate_average_error(predictions, actual_routes):
    total_error = 0
    total_points = 0
    for pred, actual in zip(predictions, actual_routes):
        if np.isnan(pred).any() or len(pred) == 0 or len(actual) == 0:
            continue
        min_length = min(len(pred), len(actual))
        error = np.mean(np.linalg.norm(pred[:min_length] - actual[:min_length], axis=1))
        total_error += error * min_length
        total_points += min_length
    return total_error / total_points if total_points > 0 else 0

class MonthlyCountPredictor(nn.Module):
    def __init__(self, input_size=12, hidden_size=24, output_size=12):
        super(MonthlyCountPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n.squeeze(0))
        return out

def prepare_monthly_count_data(data, start_year, end_year):
    monthly_counts = {year: {month: 0 for month in range(1, 13)} for year in range(start_year, end_year + 1)}
    for storm_data in data:
        year = storm_data['start_point'][0].year
        month = storm_data['month']
        if start_year <= year <= end_year:
            monthly_counts[year][month] += 1
    
    X = []
    y = []
    for year in range(start_year, end_year):
        X.append([monthly_counts[year][month] for month in range(1, 13)])
        y.append([monthly_counts[year+1][month] for month in range(1, 13)])
    
    return np.array(X), np.array(y)

def train_monthly_count_model(X, y, epochs=100):
    X_tensor = torch.FloatTensor(X).unsqueeze(1)
    y_tensor = torch.FloatTensor(y)
    
    model = MonthlyCountPredictor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return model

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Typhoon Analysis and Prediction Dashboard"),
    
    html.Div([
        dcc.Input(id='test-year', type='number', placeholder='Test Year', value=2023, min=1950, max=2024, step=1),
        html.Button('Analyze and Predict', id='analyze-button', n_clicks=0),
    ]),

    dcc.Graph(id='typhoon-routes-graph'),
    html.Div(id='fourier-results'),
    html.Div(id='prediction-results'),
    html.Div(id='monthly-typhoon-count'),
])

@app.callback(
    [Output('typhoon-routes-graph', 'figure'),
     Output('fourier-results', 'children'),
     Output('prediction-results', 'children'),
     Output('monthly-typhoon-count', 'children')],
    [Input('analyze-button', 'n_clicks')],
    [State('test-year', 'value')]
)
def update_analysis(n_clicks, test_year):
    fig_routes = go.Figure()
    route_descriptions = []

    predictions = []
    actual_routes = []
    monthly_counts = {i: 0 for i in range(1, 13)}

    # Get actual typhoon data for the test year
    test_season = ibtracs.get_season(test_year)
    for storm_id in test_season.summary()['id']:
        storm = ibtracs.get_storm(storm_id)
        lons, lats = filter_west_pacific_coordinates(np.array(storm.lon), np.array(storm.lat))
        actual_route = np.column_stack((lons, lats))
        actual_routes.append(actual_route)
        monthly_counts[storm.time[0].month] += 1

        fig_routes.add_trace(go.Scattergeo(
            lon=lons,
            lat=lats,
            mode='lines',
            name=f'Actual Storm {storm_id}',
            line=dict(color='red', width=2)
        ))

    # Predict monthly counts
    previous_year_counts = [monthly_counts.get(i, 0) for i in range(1, 13)]
    with torch.no_grad():
        predicted_counts = monthly_count_model(torch.FloatTensor(previous_year_counts).unsqueeze(0).unsqueeze(0))
    predicted_counts = predicted_counts.squeeze().numpy().round().astype(int)

    # Predict routes for each month based on predicted counts
    for month in range(1, 13):
        oni = get_oni_value(datetime(test_year, month, 1))
        for _ in range(predicted_counts[month-1]):
            try:
                # Add variability to lifetime and spawn location
                lifetime = np.random.randint(3, 14)  # Random lifetime between 3 and 14 days
                spawn_lat = np.random.uniform(5, 25)  # Random latitude between 5°N and 25°N
                spawn_lon = np.random.uniform(120, 160)  # Random longitude between 120°E and 160°E
                spawn_location = np.array([spawn_lon, spawn_lat])

                predicted_route, route_description = generate_future_route(model, scaler_X, scaler_y, month, oni, lifetime, spawn_location)
                
                if predicted_route is not None and len(predicted_route) > 0:
                    predictions.append(predicted_route)
                    fig_routes.add_trace(go.Scattergeo(
                        lon=predicted_route[:, 0],
                        lat=predicted_route[:, 1],
                        mode='lines',
                        name=f'Predicted Storm (Month {month})',
                        line=dict(color='blue', width=2)
                    ))

                    route_descriptions.append(f"Route {month}:\n{route_description}\n")
                else:
                    print(f"Warning: Invalid prediction for month {month}")
                    route_descriptions.append(f"Route {month}: Invalid prediction\n")
            except Exception as e:
                print(f"Error processing month {month}: {e}")
                route_descriptions.append(f"Route {month}: Error in prediction\n")


    fig_routes.update_layout(
        title=f'Typhoon Routes for {test_year}',
        geo=dict(
            projection_type='mercator',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            countrycolor='rgb(204, 204, 204)',
            coastlinecolor='rgb(100, 100, 100)',
            lataxis={'range': [0, 40]},
            lonaxis={'range': [100, 180]},
        )
    )

    route_results = html.Div([
        html.H3("Predicted Typhoon Routes"),
        html.Pre("\n".join(route_descriptions) if route_descriptions else "No valid route predictions generated.")
    ])

    avg_error = calculate_average_error(predictions, actual_routes)
    prediction_results = html.Div([
        html.H3(f"Prediction Results for {test_year}"),
        html.P(f"Number of predicted storms: {len(predictions)}"),
        html.P(f"Number of actual storms: {len(actual_routes)}"),
        html.P(f"Average prediction error: {avg_error:.2f} degrees")
    ])

    if monthly_count_model is not None:
        previous_year_counts = [monthly_counts.get(i, 0) for i in range(1, 13)]
        with torch.no_grad():
            predicted_counts = monthly_count_model(torch.FloatTensor(previous_year_counts).unsqueeze(0).unsqueeze(0))
        predicted_counts = predicted_counts.squeeze().numpy().round().astype(int)

        monthly_count_results = html.Div([
            html.H3(f"Monthly Typhoon Count for {test_year}"),
            html.Div([
                html.P(f"Actual: {', '.join([f'{calendar.month_abbr[m]}: {c}' for m, c in monthly_counts.items() if c > 0])}"),
                html.P(f"Predicted: {', '.join([f'{calendar.month_abbr[i+1]}: {c}' for i, c in enumerate(predicted_counts) if c > 0])}")
            ])
        ])
    else:
        monthly_count_results = html.Div([
            html.H3("Monthly Typhoon Count Prediction Not Available"),
            html.P("Please train the monthly count model first.")
        ])

    return fig_routes, route_results, prediction_results, monthly_count_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Typhoon Analysis and Prediction Dashboard')
    parser.add_argument('--start_year', type=int, default=1950, help='Start year for training data')
    parser.add_argument('--end_year', type=int, default=2022, help='End year for training data')
    parser.add_argument('--mode', choices=['train', 'predict'], default='predict', help='Mode: train or predict')
    parser.add_argument('--resume_from', type=str, help='Resume training from checkpoint file')
    parser.add_argument('--train_count_model', action='store_true', help='Train the monthly count prediction model')
    parser.add_argument('--epochs', type=int, default=1600, help='Number of epochs to train')
    args = parser.parse_args()

    ibtracs = load_ibtracs_data()
    data, max_coeff_length = prepare_data(args.start_year, args.end_year, sequence_length=10)

    if args.mode == 'train':
        model, scaler_X, scaler_y, scaler_lifetime, scaler_spawn = train_model(data, max_coeff_length, sequence_length=10, nb_epochs=args.epochs, resume_from=args.resume_from)
    
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'scaler_lifetime': scaler_lifetime,
            'scaler_spawn': scaler_spawn,
            'max_coeff_length': max_coeff_length,
            'input_size': model.get_input_size(),
            'hidden_size': model.get_hidden_size(),
            'output_size': model.get_output_size(),
            'num_coefficients': model.num_coefficients
        }, MODEL_PATH)
    
        print("Model training completed and saved.")

        if args.train_count_model:
            X, y = prepare_monthly_count_data(data, args.start_year, args.end_year)
            monthly_count_model = train_monthly_count_model(X, y)
            torch.save(monthly_count_model.state_dict(), 'monthly_count_model.pth')
            print("Monthly count model training completed and saved.")
    
    elif args.mode == 'predict':
        if os.path.exists(MODEL_PATH):
            print(f"Loading model from {MODEL_PATH}")
            checkpoint = torch.load(MODEL_PATH, map_location=device)

            input_size = checkpoint['input_size']
            hidden_size = checkpoint['hidden_size']
            num_coefficients = checkpoint['num_coefficients']

            model = MultiCoefficientPredictor(input_size, hidden_size, num_coefficients).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            scaler_X = checkpoint['scaler_X']
            scaler_y = checkpoint['scaler_y']
            scaler_lifetime = checkpoint['scaler_lifetime']
            scaler_spawn = checkpoint['scaler_spawn']
            max_coeff_length = checkpoint['max_coeff_length']

            print("Model loaded successfully.")
            print(f"Model architecture: {model}")
        else:
            print(f"No model found at {MODEL_PATH}. Please train the model first.")
            exit(1)

        if os.path.exists('monthly_count_model.pth'):
            monthly_count_model = MonthlyCountPredictor()
            monthly_count_model.load_state_dict(torch.load('monthly_count_model.pth'))
            monthly_count_model.eval()
            print("Monthly count model loaded successfully.")
        else:
            print("Monthly count model not found. Please train the model first using --train_count_model flag.")
            monthly_count_model = None

        print("Starting Dash server...")
        app.run_server(debug=True)

    else:
        print("Invalid mode. Please choose 'train' or 'predict'.")
