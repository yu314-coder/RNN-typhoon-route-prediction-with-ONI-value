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
import glob
import signal

# Constants
IBTRACS_URL = 'https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.WP.list.v04r00.csv'
IBTRACS_LOCAL_PATH = 'ibtracs_wp.csv'
MODEL_DIR = os.path.join(os.getcwd(), 'model_checkpoints')
COUNT_MODEL_PATH = os.path.join(MODEL_DIR, 'typhoon_count_model.pth')
SPAWN_MODEL_PATH = os.path.join(MODEL_DIR, 'typhoon_spawn_model.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Global variables for handling interrupts
interrupt_received = False
count_model = None
spawn_model = None
count_scaler_X = None
count_scaler_y = None
spawn_scaler_X = None
spawn_scaler_y = None
avg_daily_prob = None

def signal_handler(signum, frame):
    global interrupt_received
    print("Interrupt received. Saving models...")
    interrupt_received = True

signal.signal(signal.SIGINT, signal_handler)

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

def get_world_temperature(date):
    # This is a placeholder. Replace with actual world temperature data lookup.
    return np.random.uniform(14, 18)

def prepare_data(start_year, end_year):
    count_data = []
    spawn_data = []
    
    print("Preparing data...")
    years = list(range(start_year, end_year + 1))
    for year in tqdm(years, desc="Processing years"):
        season = ibtracs.get_season(year)
        daily_counts = {month: {day: 0 for day in range(1, 32)} for month in range(1, 13)}
        
        for storm_id in season.summary()['id']:
            storm = ibtracs.get_storm(storm_id)
            
            if isinstance(storm.time[0], datetime):
                start_date = storm.time[0]
            else:
                start_date = datetime.strptime(storm.time[0], '%Y-%m-%d %H:%M:%S')
            
            month = start_date.month
            day = start_date.day
            daily_counts[month][day] += 1
            
            spawn_location = np.array([storm.lon[0], storm.lat[0]])
            temperature = get_world_temperature(start_date)
            
            spawn_data.append({
                'spawn_location': spawn_location,
                'year': year,
                'month': month,
                'day': day,
                'dayofyear': start_date.timetuple().tm_yday,
                'temperature': temperature,
                'count': daily_counts[month][day]
            })
        
        for month in range(1, 13):
            for day in range(1, 32):
                if day <= calendar.monthrange(year, month)[1]:
                    date = datetime(year, month, day)
                    count_data.append({
                        'year': year,
                        'month': month,
                        'day': day,
                        'dayofyear': date.timetuple().tm_yday,
                        'count': daily_counts[month][day],
                        'temperature': get_world_temperature(date)
                    })
    
    return count_data, spawn_data

def calculate_avg_daily_prob(count_data):
    monthly_counts = {month: 0 for month in range(1, 13)}
    monthly_days = {month: 0 for month in range(1, 13)}
    
    for d in count_data:
        month = d['month']
        monthly_counts[month] += d['count']
        monthly_days[month] += 1
    
    monthly_probs = {}
    for month in range(1, 13):
        if monthly_days[month] > 0:
            monthly_probs[month] = monthly_counts[month] / monthly_days[month]
        else:
            monthly_probs[month] = 0
    
    print("Monthly probabilities of a typhoon:")
    for month, prob in monthly_probs.items():
        print(f"  {calendar.month_name[month]}: {prob:.4f}")
    
    total_typhoons = sum(monthly_counts.values())
    total_days = sum(monthly_days.values())
    avg_daily_prob = total_typhoons / total_days
    print(f"\nOverall average daily probability: {avg_daily_prob:.4f}")
    
    return avg_daily_prob, monthly_probs

class CountPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, avg_daily_prob, num_layers=2, dropout=0.2):
        super(CountPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.avg_daily_prob = avg_daily_prob
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        # Remove the sigmoid activation here

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch_size, seq_len, _ = x.size()
        
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        # Remove the sigmoid activation here
        return out.squeeze(-1)  # This will ensure the output is of shape [batch_size]

class SpawnPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=4, num_layers=2, dropout=0.2):
        super(SpawnPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        
        return out

def train_models(count_data, spawn_data, nb_epochs=200, batch_size=32, learning_rate=0.001):
    global count_model, spawn_model, count_scaler_X, count_scaler_y, spawn_scaler_X, spawn_scaler_y, interrupt_received, avg_daily_prob, monthly_probs
    
    avg_daily_prob, monthly_probs = calculate_avg_daily_prob(count_data)
    print(f"Average daily probability of a typhoon: {avg_daily_prob:.4f}")

    X_count = np.array([[d['year'], d['month'], d['day'], d['dayofyear'], d['temperature']] for d in count_data])
    y_count = np.array([1 if d['count'] > 0 else 0 for d in count_data])
    
    count_scaler_X = MinMaxScaler()
    X_count_scaled = count_scaler_X.fit_transform(X_count)
    
    X_count_tensor = torch.FloatTensor(X_count_scaled).to(device)
    y_count_tensor = torch.FloatTensor(y_count).to(device)
    
    count_dataset = TensorDataset(X_count_tensor, y_count_tensor)
    count_dataloader = DataLoader(count_dataset, batch_size=batch_size, shuffle=True)
    
    X_spawn = np.array([[d['year'], d['month'], d['day'], d['dayofyear'], d['temperature'], d['count']] for d in spawn_data])
    y_spawn = np.array([[d['spawn_location'][0], d['spawn_location'][1], d['month'], d['day']] for d in spawn_data])
    
    spawn_scaler_X = MinMaxScaler()
    spawn_scaler_y = MinMaxScaler()
    
    X_spawn_scaled = spawn_scaler_X.fit_transform(X_spawn)
    y_spawn_scaled = spawn_scaler_y.fit_transform(y_spawn)
    
    X_spawn_tensor = torch.FloatTensor(X_spawn_scaled).to(device)
    y_spawn_tensor = torch.FloatTensor(y_spawn_scaled).to(device)
    
    spawn_dataset = TensorDataset(X_spawn_tensor, y_spawn_tensor)
    spawn_dataloader = DataLoader(spawn_dataset, batch_size=batch_size, shuffle=True)
    
    count_model = CountPredictor(X_count.shape[1], hidden_size=64, avg_daily_prob=avg_daily_prob).to(device)
    spawn_model = SpawnPredictor(X_spawn.shape[1], hidden_size=64, output_size=4).to(device)  # output_size=4 for lon, lat, month, day
    
    count_criterion = nn.BCEWithLogitsLoss()
    spawn_criterion = nn.MSELoss()
    count_optimizer = optim.Adam(count_model.parameters(), lr=learning_rate)
    spawn_optimizer = optim.Adam(spawn_model.parameters(), lr=learning_rate)
    
    count_scheduler = optim.lr_scheduler.ReduceLROnPlateau(count_optimizer, patience=10, factor=0.5, verbose=True)
    spawn_scheduler = optim.lr_scheduler.ReduceLROnPlateau(spawn_optimizer, patience=10, factor=0.5, verbose=True)
    
    try:
        progress_bar = tqdm(total=nb_epochs, desc="Training Progress", ncols=100)
        for epoch in range(nb_epochs):
            if interrupt_received:
                break

            count_model.train()
            count_total_loss = 0
            for batch_X, batch_y in count_dataloader:
                count_optimizer.zero_grad()
                outputs = count_model(batch_X)
                loss = count_criterion(outputs, batch_y)
                loss.backward()
                count_optimizer.step()
                count_total_loss += loss.item()
            
            count_avg_loss = count_total_loss / len(count_dataloader)
            
            spawn_model.train()
            spawn_total_loss = 0
            for batch_X, batch_y in spawn_dataloader:
                spawn_optimizer.zero_grad()
                outputs = spawn_model(batch_X)
                loss = spawn_criterion(outputs, batch_y)
                loss.backward()
                spawn_optimizer.step()
                spawn_total_loss += loss.item()
            
            spawn_avg_loss = spawn_total_loss / len(spawn_dataloader)
            
            count_model.eval()
            spawn_model.eval()
            with torch.no_grad():
                count_val_loss = count_criterion(count_model(X_count_tensor), y_count_tensor).item()
                spawn_val_loss = spawn_criterion(spawn_model(X_spawn_tensor), y_spawn_tensor).item()
            
            count_scheduler.step(count_val_loss)
            spawn_scheduler.step(spawn_val_loss)
            
            progress_bar.update(1)
            
            if (epoch + 1) % 10 == 0:
                progress_bar.write(f"\nEpoch [{epoch+1}/{nb_epochs}]")
                progress_bar.write(f"Count Model - Train Loss: {count_avg_loss:.4f}, Val Loss: {count_val_loss:.4f}")
                progress_bar.write(f"Spawn Model - Train Loss: {spawn_avg_loss:.4f}, Val Loss: {spawn_val_loss:.4f}")
    
    finally:
        progress_bar.close()
        save_models(avg_daily_prob, monthly_probs)

    total_days = len(count_data)
    print(f"Average daily probability of a typhoon: {avg_daily_prob:.4f}")
    print(f"\nPercentage of data used for training: {(len(spawn_data) / total_days * 100):.2f}% for spawn model")

def save_models(avg_daily_prob, monthly_probs):
    global count_model, spawn_model, count_scaler_X, spawn_scaler_X, spawn_scaler_y
    if count_model is not None and spawn_model is not None:
        torch.save({
            'model_state_dict': count_model.state_dict(),
            'scaler_X': count_scaler_X,
            'avg_daily_prob': avg_daily_prob,
            'monthly_probs': monthly_probs
        }, COUNT_MODEL_PATH)
        
        torch.save({
            'model_state_dict': spawn_model.state_dict(),
            'scaler_X': spawn_scaler_X,
            'scaler_y': spawn_scaler_y,
        }, SPAWN_MODEL_PATH)
        
        print(f"Models, scalers, and probabilities saved to {MODEL_DIR}")

def predict_typhoon_count(year, month, day, temperature):
    date = datetime(year, month, day)
    dayofyear = date.timetuple().tm_yday
    input_features = np.array([year, month, day, dayofyear, temperature])
    scaled_input = count_scaler_X.transform(input_features.reshape(1, -1))
    
    X = torch.FloatTensor(scaled_input).to(device)
    
    count_model.eval()
    with torch.no_grad():
        pred_count = torch.sigmoid(count_model(X))  # Apply sigmoid here
    
    count = int(pred_count.item() > avg_daily_prob)
    
    print(f"Date: {year}-{month}-{day}, Predicted count: {count}")
    
    return count

def predict_spawn_location(year, month, day, temperature, nearby_counts):
    date = datetime(year, month, day)
    dayofyear = date.timetuple().tm_yday
    input_features = np.array([[year, month, day, dayofyear, temperature, nearby_counts]])
    scaled_input = spawn_scaler_X.transform(input_features)
    
    X = torch.FloatTensor(scaled_input).to(device)
    
    spawn_model.eval()
    with torch.no_grad():
        pred_spawn = spawn_model(X)
    
    pred_spawn_2d = pred_spawn.cpu().numpy()
    spawn_location = spawn_scaler_y.inverse_transform(pred_spawn_2d)[0]
    
    # Correct longitude and latitude
    corrected_lon = (spawn_location[0] + 360) % 360
    corrected_lat = abs(spawn_location[1])
    
    # Separate the output into location and date
    location = [corrected_lon, corrected_lat]
    predicted_month, predicted_day = map(int, spawn_location[2:])
    
    return location, predicted_month, predicted_day


def get_actual_storm(year, month, day):
    season = ibtracs.get_season(year)
    for storm_id in season.summary()['id']:
        storm = ibtracs.get_storm(storm_id)
        if isinstance(storm.time[0], datetime):
            start_date = storm.time[0]
        else:
            start_date = datetime.strptime(storm.time[0], '%Y-%m-%d %H:%M:%S')
        if start_date.month == month and start_date.day == day:
            return storm
    return None

def update_spawn_model(error, year, month, day, temperature, nearby_counts):
    learning_rate = 0.01
    date = datetime(year, month, day)
    dayofyear = date.timetuple().tm_yday
    input_features = np.array([[year, month, day, dayofyear, temperature, nearby_counts]])
    scaled_input = spawn_scaler_X.transform(input_features)
    
    X = torch.FloatTensor(scaled_input).to(device)
    target = torch.FloatTensor(spawn_scaler_y.transform(error.reshape(1, -1))).to(device)
    
    spawn_model.train()
    optimizer = optim.Adam(spawn_model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    optimizer.zero_grad()
    output = spawn_model(X)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Typhoon Prediction Dashboard"),
    
    html.Div([
        dcc.Input(id='test-year', type='number', placeholder='Test Year', value=2023, min=1950, max=2024, step=1),
        html.Button('Predict Typhoons', id='predict-button', n_clicks=0),
    ]),

    dcc.Graph(id='typhoon-graph'),
    html.Div(id='prediction-results'),
])

@app.callback(
    [Output('typhoon-graph', 'figure'),
     Output('prediction-results', 'children')],
    [Input('predict-button', 'n_clicks')],
    [State('test-year', 'value')]
)
def update_prediction(n_clicks, test_year):
    fig = go.Figure()
    prediction_results = []
    all_predictions = []

    all_counts = {}
    for month in tqdm(range(1, 13), desc="Predicting counts"):
        all_counts[month] = {}
        for day in range(1, calendar.monthrange(test_year, month)[1] + 1):
            date = datetime(test_year, month, day)
            temperature = get_world_temperature(date)
            count = predict_typhoon_count(test_year, month, day, temperature)
            all_counts[month][day] = count

    for month in tqdm(range(1, 13), desc="Predicting spawns"):
        for day in range(1, calendar.monthrange(test_year, month)[1] + 1):
            date = datetime(test_year, month, day)
            temperature = get_world_temperature(date)
            
            nearby_days = range(max(1, day - 3), min(calendar.monthrange(test_year, month)[1], day + 4))
            nearby_counts = sum(all_counts[month].get(d, 0) for d in nearby_days)
            
            predicted_count = all_counts[month][day]
            
            if predicted_count > 0:
                location, pred_month, pred_day = predict_spawn_location(test_year, month, day, temperature, nearby_counts)
                
                # Store all predictions
                all_predictions.append(f"Date: {date.strftime('%Y-%m-%d')}, Predicted location: ({location[0]:.2f}°E, {location[1]:.2f}°N), Count: {predicted_count}, Temp: {temperature:.2f}°C")
                
                # Only add points within the Western Pacific region to the map
                if 100 <= location[0] <= 180 and 0 <= location[1] <= 40:
                    fig.add_trace(go.Scattergeo(
                        lon=[location[0]],
                        lat=[location[1]],
                        mode='markers',
                        name=f'Typhoon ({date.strftime("%Y-%m-%d")})',
                        marker=dict(size=10, color=month, colorscale='Viridis', showscale=True, colorbar=dict(title='Month'))
                    ))
                
                    prediction_results.append(f"Date: {date.strftime('%Y-%m-%d')}, Count: {predicted_count}, Spawn: ({location[0]:.2f}°E, {location[1]:.2f}°N), Predicted Date: {pred_month}/{pred_day}, Temp: {temperature:.2f}°C")

    fig.update_layout(
        title=f'Predicted Typhoon Spawn Points for {test_year}',
        geo=dict(
            projection_type='mercator',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            countrycolor='rgb(204, 204, 204)',
            coastlinecolor='rgb(100, 100, 100)',
            lataxis={'range': [0, 40]},
            lonaxis={'range': [100, 180]},
            center=dict(lon=140, lat=20),
            projection_scale=2
        )
    )

    # Print all predictions
    print("\nAll Predictions:")
    for pred in all_predictions:
        print(pred)

    if not prediction_results:
        prediction_results.append("No typhoons predicted within the Western Pacific region (100°E to 180°E and 0°N to 40°N).")

    return fig, html.Ul([html.Li(result) for result in prediction_results])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Typhoon Prediction')
    parser.add_argument('--start_year', type=int, default=1950, help='Start year for training data')
    parser.add_argument('--end_year', type=int, default=2022, help='End year for training data')
    parser.add_argument('--mode', choices=['train', 'predict'], default='predict', help='Mode: train or predict')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    args = parser.parse_args()

    if args.mode == 'train':
        print("Preparing data...")
        count_data, spawn_data = prepare_data(args.start_year, args.end_year)
        train_models(count_data, spawn_data, nb_epochs=args.epochs)
        
    elif args.mode == 'predict':
        if os.path.exists(COUNT_MODEL_PATH) and os.path.exists(SPAWN_MODEL_PATH):
            print(f"Loading models from {COUNT_MODEL_PATH} and {SPAWN_MODEL_PATH}")
            count_checkpoint = torch.load(COUNT_MODEL_PATH, map_location=device)
            spawn_checkpoint = torch.load(SPAWN_MODEL_PATH, map_location=device)
          
            avg_daily_prob = count_checkpoint['avg_daily_prob']
            monthly_probs = count_checkpoint['monthly_probs']
    
            print("Loaded monthly probabilities of a typhoon:")
            for month, prob in monthly_probs.items():
                print(f"  {calendar.month_name[month]}: {prob:.4f}")
            print(f"\nLoaded overall average daily probability: {avg_daily_prob:.4f}")

            count_input_size = 5  # year, month, day, dayofyear, temperature
            spawn_input_size = 6  # year, month, day, dayofyear, temperature, nearby_counts
            hidden_size = 64

            count_model = CountPredictor(count_input_size, hidden_size, avg_daily_prob).to(device)
            count_model.load_state_dict(count_checkpoint['model_state_dict'])
            count_model.eval()

            spawn_model = SpawnPredictor(spawn_input_size, hidden_size, output_size=4).to(device)
            
            # Handle the structural change in SpawnPredictor
            old_state_dict = spawn_checkpoint['model_state_dict']
            new_state_dict = spawn_model.state_dict()

            # Copy over the LSTM weights
            for key in ['lstm.weight_ih_l0', 'lstm.weight_hh_l0', 'lstm.bias_ih_l0', 'lstm.bias_hh_l0']:
                if key in old_state_dict:
                    new_state_dict[key] = old_state_dict[key]

            print("Loaded LSTM weights from saved model. FC layer initialized with random weights.")
            spawn_model.load_state_dict(new_state_dict)
            spawn_model.eval()

            count_scaler_X = count_checkpoint['scaler_X']
            spawn_scaler_X = spawn_checkpoint['scaler_X']
            spawn_scaler_y = spawn_checkpoint['scaler_y']

            print("Models loaded successfully.")

            print("Starting Dash server...")
            app.run_server(debug=True)
        else:
            print(f"No models found at {COUNT_MODEL_PATH} or {SPAWN_MODEL_PATH}. Please train the models first.")
            exit(1)

    else:
        print("Invalid mode. Please choose 'train' or 'predict'.")