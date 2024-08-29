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

# Constants
CHECKPOINT_DIR = 'checkpoints'
CHECKPOINT_INTERVAL = 1000
FOURIER_DATA_DIR = 'fourier_data'
IBTRACS_URL = 'https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.WP.list.v04r00.csv'
IBTRACS_LOCAL_PATH = 'ibtracs_wp.csv'
ONI_DATA_PATH = os.path.join(os.getcwd(), 'oni_data.csv')
MODEL_PATH = os.path.join(os.getcwd(), 'hurricane_rnn_model.pth')
SCALER_PATH = os.path.join(os.getcwd(), 'scaler.joblib')

# Load IBTrACS data
ibtracs = tracks.TrackDataset(basin='west_pacific', source='ibtracs')

def update_ibtracs_data():
    if not os.path.exists(IBTRACS_LOCAL_PATH) or (datetime.now() - datetime.fromtimestamp(os.path.getmtime(IBTRACS_LOCAL_PATH)) > timedelta(days=1)):
        print("Updating IBTrACS data...")
        response = requests.get(IBTRACS_URL)
        with open(IBTRACS_LOCAL_PATH, 'wb') as f:
            f.write(response.content)
        print("IBTrACS data updated.")
    else:
        print("IBTrACS data is up to date.")

def filter_west_pacific_coordinates(lons, lats):
    mask = (100 <= lons) & (lons <= 180) & (0 <= lats) & (lats <= 40)
    return lons[mask], lats[mask]

def compute_fourier_coefficients(lons, lats, n_components=10):
    t = np.linspace(0, 2*np.pi, len(lons))
    coeffs_lon = np.fft.rfft(lons)[:n_components+1]
    coeffs_lat = np.fft.rfft(lats)[:n_components+1]
    
    coeffs_lon = np.pad(coeffs_lon, (0, max(0, n_components+1 - len(coeffs_lon))))[:n_components+1]
    coeffs_lat = np.pad(coeffs_lat, (0, max(0, n_components+1 - len(coeffs_lat))))[:n_components+1]
    
    return np.concatenate([coeffs_lon.real, coeffs_lon.imag, coeffs_lat.real, coeffs_lat.imag])

def inverse_fourier_transform(coeffs, n_points=100):
    n_components = (len(coeffs) // 4) - 1
    coeffs_lon = coeffs[:n_components+1] + 1j * coeffs[n_components+1:2*n_components+2]
    coeffs_lat = coeffs[2*n_components+2:3*n_components+3] + 1j * coeffs[3*n_components+3:]
    t = np.linspace(0, 2*np.pi, n_points)
    lons = np.fft.irfft(coeffs_lon, n=n_points)
    lats = np.fft.irfft(coeffs_lat, n=n_points)
    return lons, lats

def prepare_data(start_year, end_year, sequence_length, n_fourier_components=10):
    update_ibtracs_data()
    
    data_filename = f'typhoon_data_{start_year}_{end_year}.npz'
    
    if os.path.exists(data_filename):
        print("Loading existing typhoon data...")
        data = np.load(data_filename, allow_pickle=True)
        return data['X'], data['y'], data['start_points'], data['lifetimes'], data['spawn_locations']
    
    print("Preparing typhoon data...")
    X, y, start_points, lifetimes, spawn_locations = [], [], [], [], []
    
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
            X.append(fourier_coeffs)
            y.append(np.array([lons[-1], lats[-1]]))
            start_points.append(np.array([lons[0], lats[0]]))
            lifetimes.append(lifetime)
            spawn_locations.append(np.array([lons[0], lats[0]]))
    
    X = np.array(X)
    y = np.array(y)
    start_points = np.array(start_points)
    lifetimes = np.array(lifetimes)
    spawn_locations = np.array(spawn_locations)
    
    np.savez(data_filename, X=X, y=y, start_points=start_points, lifetimes=lifetimes, spawn_locations=spawn_locations)
    
    return X, y, start_points, lifetimes, spawn_locations

class HurricaneRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, fc_start_input_size, num_layers=2, dropout=0.2):
        super(HurricaneRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size + 3, output_size)
        self.fc_start = nn.Linear(fc_start_input_size, input_size)

    def forward(self, x, lifetime, spawn_location):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch_size, seq_len, _ = x.size()
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        if x.size(-1) == 2:
            x = torch.cat([x.squeeze(1), lifetime.unsqueeze(1), spawn_location], dim=1)
            x = self.fc_start(x).unsqueeze(1)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        
        if self.training:
            out = self.bn(out)
        
        out = torch.cat([out, lifetime.unsqueeze(1), spawn_location], dim=1)
        out = self.fc(out)
        return out

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

def load_checkpoint(filename):
    path = os.path.join(CHECKPOINT_DIR, filename)
    if os.path.exists(path):
        return torch.load(path)
    return None

def train_model(X, y, start_points, lifetimes, spawn_locations, sequence_length, nb_epochs=150, batch_size=32, learning_rate=0.001, resume_from=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    
    input_size = X.shape[1]
    hidden_size = 256
    output_size = 2
    fc_start_input_size = 5

    model = HurricaneRNN(input_size, hidden_size, output_size, fc_start_input_size).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)
    
    start_epoch = 0
    start_batch = 0
    
    if resume_from:
        checkpoint = load_checkpoint(resume_from)
        if checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            start_batch = checkpoint['batch']
            scaler_X = checkpoint['X_scaler']
            scaler_y = checkpoint['y_scaler']
            scaler_start = checkpoint['start_scaler']
            scaler_lifetime = checkpoint['lifetime_scaler']
            scaler_spawn = checkpoint['spawn_scaler']
            print(f"Resuming from epoch {start_epoch}, batch {start_batch}")
    
    for epoch in tqdm(range(start_epoch, nb_epochs)):
        model.train()
        total_loss = 0
        for batch, (batch_X, batch_y, batch_start, batch_lifetime, batch_spawn) in enumerate(dataloader, start=start_batch):
            optimizer.zero_grad()
            outputs = model(batch_X, batch_lifetime, batch_spawn)
            loss = criterion(outputs, batch_y)
            
            predicted_start = model(batch_start, batch_lifetime, batch_spawn)
            start_loss = criterion(predicted_start, batch_start)
            loss += 0.1 * start_loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if (batch + 1) % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(model, optimizer, epoch, batch, scaler_X, scaler_y, scaler_start, scaler_lifetime, scaler_spawn, f'checkpoint_epoch{epoch}_batch{batch}.pt')
        
        avg_loss = total_loss / len(dataloader)
        
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_tensor, lifetimes_tensor, spawn_locations_tensor), y_tensor).item()
        
        scheduler.step(val_loss)
        
        print(f"Epoch [{epoch+1}/{nb_epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        save_checkpoint(model, optimizer, epoch + 1, 0, scaler_X, scaler_y, scaler_start, scaler_lifetime, scaler_spawn, f'checkpoint_epoch{epoch+1}.pt')
    
    return model, scaler_X, scaler_y, scaler_lifetime, scaler_spawn

def generate_future_route(model, scaler_X, scaler_y, scaler_lifetime, scaler_spawn, start_point, lifetime, spawn_location, max_steps=100, n_points=200):
    device = next(model.parameters()).device
    route = [start_point]
    fourier_series = []
    current_sequence = np.array([start_point])
    
    for _ in range(min(lifetime, max_steps)):
        if len(current_sequence) < 20:
            padded_sequence = np.pad(current_sequence, ((0, 20 - len(current_sequence)), (0, 0)), mode='edge')
        else:
            padded_sequence = current_sequence[-20:]
        
        fourier_coeffs = compute_fourier_coefficients(padded_sequence[:, 0], padded_sequence[:, 1])
        fourier_coeffs_scaled = scaler_X.transform(fourier_coeffs.reshape(1, -1))
        lifetime_scaled = scaler_lifetime.transform([[lifetime]])[0][0]
        spawn_location_scaled = scaler_spawn.transform(spawn_location.reshape(1, -1))
        
        input_tensor = torch.FloatTensor(fourier_coeffs_scaled).to(device)
        lifetime_tensor = torch.FloatTensor([lifetime_scaled]).to(device)
        spawn_location_tensor = torch.FloatTensor(spawn_location_scaled).to(device)
        
        model.eval()
        with torch.no_grad():
            prediction_scaled = model(input_tensor, lifetime_tensor, spawn_location_tensor)
        
        prediction = scaler_y.inverse_transform(prediction_scaled.cpu().numpy())
        next_point = prediction[0]
        
        next_point[0] = np.clip(next_point[0], 100, 180)  # Longitude
        next_point[1] = np.clip(next_point[1], 0, 40)    # Latitude
        
        route.append(next_point)
        fourier_series.append(fourier_coeffs)
        current_sequence = np.vstack((current_sequence, next_point))
        
        if np.linalg.norm(next_point - start_point) > 20:
            break
    
    t = np.linspace(0, 2*np.pi, n_points)
    smooth_route = np.zeros((n_points, 2))
    for coeffs in fourier_series:
        lons, lats = inverse_fourier_transform(coeffs, n_points)
        smooth_route += np.column_stack((lons, lats))
    smooth_route /= len(fourier_series)
    
    smooth_route[:, 0] = np.clip(smooth_route[:, 0], 100, 180)  # Longitude
    smooth_route[:, 1] = np.clip(smooth_route[:, 1], 0, 40)     # Latitude
    
    return np.array(fourier_series), smooth_route

def calculate_average_error(predictions):
    total_error = 0
    total_points = 0
    for pred in predictions:
        min_length = min(len(pred['predicted']), len(pred['actual']))
        error = np.mean(np.linalg.norm(pred['predicted'][:min_length] - pred['actual'][:min_length], axis=1))
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

def prepare_monthly_count_data(start_year, end_year):
    monthly_counts = {year: {month: 0 for month in range(1, 13)} for year in range(start_year, end_year + 1)}
    for year in range(start_year, end_year + 1):
        season = ibtracs.get_season(year)
        for storm_id in season.summary()['id']:
            storm = ibtracs.get_storm(storm_id)
            if isinstance(storm.time[0], datetime):
                month = storm.time[0].month
            else:
                month = datetime.strptime(storm.time[0], '%Y-%m-%d %H:%M:%S').month
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Typhoon Analysis and Prediction Dashboard', add_help=False)
    parser.add_argument('--custom-help', action='store_true', help='Show this help message and exit after running the script')
    parser.add_argument('--start_year', type=int, default=1950, help='Start year for training data')
    parser.add_argument('--end_year', type=int, default=2022, help='End year for training data')
    parser.add_argument('--mode', choices=['train', 'predict'], default='predict', help='Mode: train or predict')
    parser.add_argument('--resume_from', type=str, help='Resume training from checkpoint file')
    parser.add_argument('--train_count_model', action='store_true', help='Train the monthly count prediction model')
    
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='Show this help message and exit')
    
    args, unknown = parser.parse_known_args()

    print("Running the main script...")

    if args.custom_help:
        print("\nCustom Help Information:")
        parser.print_help()
        sys.exit(0)

    if args.mode == 'train':
        print("Preparing data...")
        X, y, start_points, lifetimes, spawn_locations = prepare_data(args.start_year, args.end_year, sequence_length=10)
        
        print("Training model...")
        model, scaler_X, scaler_y, scaler_lifetime, scaler_spawn = train_model(X, y, start_points, lifetimes, spawn_locations, sequence_length=10, resume_from=args.resume_from)
        print("Model training completed.")
        
        torch.save(model.state_dict(), MODEL_PATH)
        joblib.dump((scaler_X, scaler_y, scaler_lifetime, scaler_spawn), SCALER_PATH)
    elif args.mode == 'predict':
        checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pt')]
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda x: os.path.getctime(os.path.join(CHECKPOINT_DIR, x)))
            checkpoint_path = os.path.join(CHECKPOINT_DIR, latest_checkpoint)
            print(f"Loading latest checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            
            input_size = checkpoint['model_state_dict']['lstm.weight_ih_l0'].shape[1]
            hidden_size = checkpoint['model_state_dict']['lstm.weight_hh_l0'].shape[0] // 4
            output_size = checkpoint['model_state_dict']['fc.weight'].shape[0]
            fc_start_input_size = checkpoint['model_state_dict']['fc_start.weight'].shape[1]
        
            model = HurricaneRNN(input_size, hidden_size, output_size, fc_start_input_size)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
        
            scaler_X = checkpoint['X_scaler']
            scaler_y = checkpoint['y_scaler']
            scaler_lifetime = checkpoint['lifetime_scaler']
            scaler_spawn = checkpoint['spawn_scaler']
            print("Model loaded successfully.")
        
            n_fourier_components = (input_size - 3) // 4
        else:
            print(f"No checkpoint files found in {CHECKPOINT_DIR}. Please train the model first.")
            exit(1)

    if args.train_count_model:
        print("Preparing monthly count data...")
        X_count, y_count = prepare_monthly_count_data(1950, 2022)
        print("Training monthly count model...")
        monthly_count_model = train_monthly_count_model(X_count, y_count)
        torch.save(monthly_count_model.state_dict(), 'monthly_count_model.pth')
        print("Monthly count model saved.")
    else:
        if os.path.exists('monthly_count_model.pth'):
            monthly_count_model = MonthlyCountPredictor()
            monthly_count_model.load_state_dict(torch.load('monthly_count_model.pth', weights_only=True))
            monthly_count_model.eval()
            print("Monthly count model loaded successfully.")
        else:
            print("Monthly count model not found. Please train the model first using --train_count_model flag.")
            monthly_count_model = None

    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1("Typhoon Analysis and Prediction Dashboard"),
        
        html.Div([
            dcc.Input(id='test-year', type='number', placeholder='Test Year', value=2023, min=1950, max=2024, step=1),
            html.Button('Analyze and Predict', id='analyze-button', n_clicks=0),
        ]),

        dcc.Graph(id='typhoon-routes-graph'),
        dcc.Graph(id='fourier-series-graph'),
        html.Div(id='prediction-results'),
        html.Div(id='monthly-typhoon-count'),
    ])

    @app.callback(
        [Output('typhoon-routes-graph', 'figure'),
         Output('fourier-series-graph', 'figure'),
         Output('prediction-results', 'children'),
         Output('monthly-typhoon-count', 'children')],
        [Input('analyze-button', 'n_clicks')],
        [State('test-year', 'value')]
    )
    def update_analysis(n_clicks, test_year):
        fig_routes = go.Figure()
        fig_fourier = go.Figure()

        test_season = ibtracs.get_season(test_year)
        predictions = []
        monthly_counts = {i: 0 for i in range(1, 13)}

        for storm_id in test_season.summary()['id']:
            storm = ibtracs.get_storm(storm_id)
            lons, lats = filter_west_pacific_coordinates(np.array(storm.lon), np.array(storm.lat))

            if len(lons) < 11:
                continue

            if isinstance(storm.time[0], datetime):
                storm_start_date = storm.time[0]
                storm_end_date = storm.time[-1]
            else:
                storm_start_date = datetime.strptime(storm.time[0], '%Y-%m-%d %H:%M:%S')
                storm_end_date = datetime.strptime(storm.time[-1], '%Y-%m-%d %H:%M:%S')

            lifetime = (storm_end_date - storm_start_date).days + 1

            month = storm_start_date.month
            monthly_counts[month] += 1

            input_sequence = np.column_stack((lons[:10], lats[:10]))
            spawn_location = np.array([lons[0], lats[0]])
            fourier_series, predicted_route = generate_future_route(model, scaler_X, scaler_y, scaler_lifetime, scaler_spawn, input_sequence[0], lifetime, spawn_location, max_steps=lifetime)

            predictions.append({
                'input': input_sequence,
                'predicted': predicted_route,
                'actual': np.column_stack((lons[10:], lats[10:])),
                'fourier': fourier_series
            })
            
            fig_routes.add_trace(go.Scattergeo(
                lon=predicted_route[:, 0],
                lat=predicted_route[:, 1],
                mode='lines',
                name=f'Predicted Storm {storm_id}',
                line=dict(color='blue', width=2)
            ))
            fig_routes.add_trace(go.Scattergeo(
                lon=lons,
                lat=lats,
                mode='lines',
                name=f'Actual Storm {storm_id}',
                line=dict(color='red', width=2)
            ))

            for i in range(11):
                fig_fourier.add_trace(go.Scatter(
                    y=fourier_series[:, i],
                    mode='lines',
                    name=f'a{i} (cos) - Longitude'
                ))
                fig_fourier.add_trace(go.Scatter(
                    y=fourier_series[:, 11+i],
                    mode='lines',
                    name=f'b{i} (sin) - Longitude'
                ))
                fig_fourier.add_trace(go.Scatter(
                    y=fourier_series[:, 22+i],
                    mode='lines',
                    name=f'a{i} (cos) - Latitude'
                ))
                fig_fourier.add_trace(go.Scatter(
                    y=fourier_series[:, 33+i],
                    mode='lines',
                    name=f'b{i} (sin) - Latitude'
                ))

        fig_routes.update_layout(
            title=f'Typhoon Analysis for {test_year}',
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

        fig_fourier.update_layout(
            title=f'Fourier Series Coefficients for Predicted Routes ({test_year})',
            xaxis_title='Time Step',
            yaxis_title='Coefficient Value',
            legend_title='Coefficient',
            height=800
        )

        prediction_results = html.Div([
            html.H3(f"Prediction Results for {test_year}"),
            html.P(f"Number of storms predicted: {len(predictions)}"),
            html.P(f"Average prediction error: {calculate_average_error(predictions):.2f} degrees")
        ])

        if monthly_count_model is not None:
            previous_year_counts = np.array([monthly_counts.get(i, 0) for i in range(1, 13)])
            with torch.no_grad():
                predicted_counts = monthly_count_model(torch.FloatTensor(previous_year_counts).unsqueeze(0).unsqueeze(0))
            predicted_counts = predicted_counts.squeeze().numpy().round().astype(int)

            monthly_count_results = html.Div([
                html.H3(f"Predicted Monthly Typhoon Count for {test_year}"),
                html.Ul([html.Li(f"{calendar.month_name[i+1]}: {count}") for i, count in enumerate(predicted_counts) if count > 0])
            ])
        else:
            monthly_count_results = html.Div([
                html.H3("Monthly Typhoon Count Prediction Not Available"),
                html.P("Please train the monthly count model first using the --train_count_model flag.")
            ])

        return fig_routes, fig_fourier, prediction_results, monthly_count_results

    print("Starting Dash server...")
    app.run_server(debug=True)
