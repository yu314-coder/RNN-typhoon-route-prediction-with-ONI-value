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
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset
import threading
import time

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



interrupt_received = False
user_input_received = threading.Event()
should_exit = threading.Event()

def signal_handler(signum, frame):
    global interrupt_received
    interrupt_received = True
    should_exit.set()
    print("\nInterrupt received. Do you want to save the models? (y/n)")
    
    def input_thread():
        global user_input
        user_input = input().lower()
        user_input_received.set()

    input_thread = threading.Thread(target=input_thread)
    input_thread.daemon = True
    input_thread.start()

    input_thread.join(timeout=30)

    if user_input_received.is_set():
        if user_input == 'y':
            print("Saving models...")
            save_models(avg_daily_prob, monthly_probs)
        print("Exiting the program.")
    else:
        print("\nNo input received. Automatically saving the models...")
        save_models(avg_daily_prob, monthly_probs)
        print("Exiting the program.")
    
    os._exit(0)  # Force exit the program

def save_timer():
    for i in range(30, 0, -1):
        if user_input_received.is_set():
            return
        print(f"\rTime remaining to answer: {i} seconds", end="", flush=True)
        time.sleep(1)
    
    if not user_input_received.is_set():
        print("\nNo input received. Automatically saving the models...")
        save_models(avg_daily_prob, monthly_probs)
        print("Exiting the program.")
        sys.exit(0)


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
    
    # Convert data to time series format
    count_ts = to_time_series_dataset([[d['count']] for d in count_data])
    spawn_ts = to_time_series_dataset([[d['spawn_location'][0], d['spawn_location'][1]] for d in spawn_data])
    
    # Normalize time series data
    count_scaler = TimeSeriesScalerMeanVariance()
    spawn_scaler = TimeSeriesScalerMeanVariance()
    count_ts_normalized = count_scaler.fit_transform(count_ts)
    spawn_ts_normalized = spawn_scaler.fit_transform(spawn_ts)
    
    # Update count_data and spawn_data with normalized values
    for i, d in enumerate(count_data):
        d['count_normalized'] = count_ts_normalized[i, 0, 0]
    
    for i, d in enumerate(spawn_data):
        d['spawn_location_normalized'] = spawn_ts_normalized[i, 0]
        d['count_normalized'] = count_ts_normalized[i % len(count_data), 0, 0]
        
        # Normalize month and temperature
        d['month_normalized'] = (d['month'] - 1) / 11  # 0 to 1
        d['temperature_normalized'] = (d['temperature'] - 14) / (18 - 14)  # Assuming temperature range 14 to 18
    
    return count_data, spawn_data

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                param.data.fill_(0.01)

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
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.avg_daily_prob = avg_daily_prob
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

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
        
        return out.squeeze(-1)


class SpawnPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=4, num_layers=2, dropout=0.2):
        super(SpawnPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
    
        batch_size = x.size(0)
    
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
    
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
    
        # Strictly enforce longitude range: 100E to 180E
        lon = self.sigmoid(out[:, 0]) * 80 + 100
    
        # Strictly enforce latitude range: 0N to 40N
        lat = self.sigmoid(out[:, 1]) * 40
    
        # Month and temperature influence on lon/lat
        month = x[:, 0, 1]  # Assuming month is the second input feature
        temp = x[:, 0, 4]   # Assuming temperature is the fifth input feature
    
        # Adjust longitude based on month and temperature
        lon = lon + (month - 0.5) * 10 + (temp - 0.5) * 5
        lon = torch.clamp(lon, 100, 180)
    
        # Adjust latitude based on month and temperature
        lat = lat + (month - 0.5) * 5 + (temp - 0.5) * 2.5
        lat = torch.clamp(lat, 0, 40)
    
        # Combine the outputs
        return torch.stack([lon, lat, out[:, 2], out[:, 3]], dim=1)

def train_models(count_data, spawn_data, nb_epochs=200, batch_size=32, learning_rate=0.001, start_epoch=0):
    global count_model, spawn_model, count_scaler_X, count_scaler_y, spawn_scaler_X, spawn_scaler_y, interrupt_received, avg_daily_prob, monthly_probs
    torch.autograd.set_detect_anomaly(True)

    X_count = np.array([[d['year'], d['month'], d['day'], d['dayofyear'], d['temperature'], d['count_normalized']] for d in count_data])
    y_count = np.array([1 if d['count'] > 0 else 0 for d in count_data])
    
    X_spawn = np.array([[d['year'], d['month_normalized'], d['day'], d['dayofyear'], d['temperature_normalized'], d['count_normalized']] for d in spawn_data])
    y_spawn = np.array([[d['spawn_location'][0], d['spawn_location'][1], d['month'], d['day']] for d in spawn_data])

    # Initialize models
    count_model = CountPredictor(X_count.shape[1], hidden_size=64, avg_daily_prob=0.5).to(device)
    spawn_model = SpawnPredictor(X_spawn.shape[1], hidden_size=64, output_size=4).to(device)
    
    # Initialize or load scalers
    count_scaler_X = MinMaxScaler()
    spawn_scaler_X = MinMaxScaler()
    spawn_scaler_y = MinMaxScaler()
    
    # Load previous training progress if available
    other_data_path = os.path.join(MODEL_DIR, 'other_data.pkl')
    if os.path.exists(COUNT_MODEL_PATH) and os.path.exists(SPAWN_MODEL_PATH) and os.path.exists(other_data_path):
        print("Loading previous training progress...")
        count_model.load_state_dict(torch.load(COUNT_MODEL_PATH, map_location=device))
        spawn_model.load_state_dict(torch.load(SPAWN_MODEL_PATH, map_location=device))
        
        with open(other_data_path, 'rb') as f:
            other_data = pickle.load(f)
        
        count_scaler_X = other_data['count_scaler_X']
        spawn_scaler_X = other_data['spawn_scaler_X']
        spawn_scaler_y = other_data['spawn_scaler_y']
        avg_daily_prob = other_data['avg_daily_prob']
        monthly_probs = other_data['monthly_probs']
        
        count_model.avg_daily_prob = avg_daily_prob
        
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print("Starting new training session")
        avg_daily_prob, monthly_probs = calculate_avg_daily_prob(count_data)
        count_model.avg_daily_prob = avg_daily_prob
    
    print(f"Average daily probability of a typhoon: {avg_daily_prob:.4f}")

    # Scale the input data
    X_count_scaled = count_scaler_X.fit_transform(X_count)
    X_spawn_scaled = spawn_scaler_X.fit_transform(X_spawn)
    y_spawn_scaled = spawn_scaler_y.fit_transform(y_spawn)
    
    # Convert to PyTorch tensors
    X_count_tensor = torch.FloatTensor(X_count_scaled).to(device)
    y_count_tensor = torch.FloatTensor(y_count).to(device)
    X_spawn_tensor = torch.FloatTensor(X_spawn_scaled).to(device)
    y_spawn_tensor = torch.FloatTensor(y_spawn_scaled).to(device)
    
    # Create data loaders
    count_dataset = TensorDataset(X_count_tensor, y_count_tensor)
    count_dataloader = DataLoader(count_dataset, batch_size=batch_size, shuffle=True)
    spawn_dataset = TensorDataset(X_spawn_tensor, y_spawn_tensor)
    spawn_dataloader = DataLoader(spawn_dataset, batch_size=batch_size, shuffle=True)
    
    # Define loss functions and optimizers
    count_criterion = nn.BCEWithLogitsLoss()
    spawn_criterion = nn.MSELoss()
    count_optimizer = optim.Adam(count_model.parameters(), lr=learning_rate)
    spawn_optimizer = optim.Adam(spawn_model.parameters(), lr=learning_rate)
    
    count_scheduler = optim.lr_scheduler.ReduceLROnPlateau(count_optimizer, patience=10, factor=0.5, verbose=True)
    spawn_scheduler = optim.lr_scheduler.ReduceLROnPlateau(spawn_optimizer, patience=10, factor=0.5, verbose=True)
    
    try:
        progress_bar = tqdm(total=nb_epochs, desc="Training Progress", ncols=100, initial=start_epoch)
        for epoch in range(start_epoch, nb_epochs):
            if should_exit.is_set():
                print("Interrupt detected, stopping training...")
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
                
                # Positive feedback: Reward accurate predictions
                with torch.no_grad():
                    lon_accuracy = 1 - torch.abs(outputs[:, 0] - batch_y[:, 0]) / 80  # Normalized by longitude range
                    lat_accuracy = 1 - torch.abs(outputs[:, 1] - batch_y[:, 1]) / 40  # Normalized by latitude range
                    accuracy = (lon_accuracy + lat_accuracy) / 2
                    positive_feedback = torch.exp(-accuracy)

                adjusted_loss = loss * positive_feedback.mean().item()  # Use .item() to get a scalar value
                
                adjusted_loss.backward()
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

            if should_exit.is_set():
                print("Interrupt detected, stopping training...")
                break

    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        progress_bar.close()

    if not should_exit.is_set():
        total_days = len(count_data)
        print(f"Average daily probability of a typhoon: {avg_daily_prob:.4f}")
        print(f"\nPercentage of data used for training: {(len(spawn_data) / total_days * 100):.2f}% for spawn model")
        
        print("Training completed. Saving models...")
        save_models(avg_daily_prob, monthly_probs)

def save_models(avg_daily_prob, monthly_probs):
    global count_model, spawn_model, count_scaler_X, spawn_scaler_X, spawn_scaler_y
    if count_model is not None and spawn_model is not None:
        # Save model state dictionaries
        torch.save(count_model.state_dict(), COUNT_MODEL_PATH)
        torch.save(spawn_model.state_dict(), SPAWN_MODEL_PATH)
        
        # Save other data
        other_data = {
            'count_scaler_X': count_scaler_X,
            'spawn_scaler_X': spawn_scaler_X,
            'spawn_scaler_y': spawn_scaler_y,
            'avg_daily_prob': avg_daily_prob,
            'monthly_probs': monthly_probs
        }
        with open(os.path.join(MODEL_DIR, 'other_data.pkl'), 'wb') as f:
            pickle.dump(other_data, f)
        
        print(f"Models and data saved to {MODEL_DIR}")
    else:
        print("Models not initialized. Nothing to save.")

def predict_typhoon_count(year, month):
    prob = monthly_probs[month]
    days_in_month = calendar.monthrange(year, month)[1]
    expected_count = round(prob * days_in_month)
    return expected_count

def predict_spawn_location(year, month, temperature):
    # Normalize month and temperature
    month_normalized = (month - 1) / 11
    temperature_normalized = (temperature - 14) / (18 - 14)
    
    input_features = np.array([[year, month_normalized, 15, (month-1)*30 + 15, temperature_normalized, 0]])  # Use middle of month for day and dayofyear
    scaled_input = spawn_scaler_X.transform(input_features)
    
    X = torch.FloatTensor(scaled_input).to(device)
    
    spawn_model.eval()
    with torch.no_grad():
        pred_spawn = spawn_model(X)
    
    pred_spawn_2d = pred_spawn.cpu().numpy()[0]
    
    # Adjust longitude based on month
    longitude_base = 100 + (month / 12) * 80  # Shift eastward as the year progresses
    longitude = np.clip(longitude_base + pred_spawn_2d[0] * 20, 100, 180)
    
    # Adjust latitude based on month
    latitude_base = 20 + np.sin((month - 1) * np.pi / 6) * 10  # Seasonal north-south movement
    latitude = np.clip(latitude_base + pred_spawn_2d[1] * 10, 0, 40)
    
    # Add random noise to predictions
    longitude = np.clip(longitude + np.random.normal(0, 2), 100, 180)
    latitude = np.clip(latitude + np.random.normal(0, 1), 0, 40)
    
    return [longitude, latitude]

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
    if n_clicks == 0:
        return go.Figure(), "Click the button to predict typhoons."

    fig = go.Figure()
    prediction_results = []

    for month in range(1, 13):
        expected_count = predict_typhoon_count(test_year, month)
        temperature = get_world_temperature(datetime(test_year, month, 15))  # Use middle of month
        
        # Generate multiple potential spawn locations
        potential_spawns = []
        for _ in range(max(expected_count * 3, 10)):  # Generate at least 10 potential spawns
            location = predict_spawn_location(test_year, month, temperature)
            potential_spawns.append(location)
        
        # Sort potential spawns by some criteria (e.g., distance from typical spawn areas)
        typical_spawn = [130, 20]  # Example: 130°E, 20°N
        potential_spawns.sort(key=lambda x: ((x[0] - typical_spawn[0])**2 + (x[1] - typical_spawn[1])**2)**0.5)
        
        # Select the top 'expected_count' spawns
        selected_spawns = potential_spawns[:expected_count]
        
        for location in selected_spawns:
            if 100 <= location[0] <= 180 and 0 <= location[1] <= 40:
                fig.add_trace(go.Scattergeo(
                    lon=[location[0]],
                    lat=[location[1]],
                    mode='markers',
                    name=f'Typhoon ({test_year}-{month:02d})',
                    marker=dict(size=10, color=month, colorscale='Viridis', showscale=True, colorbar=dict(title='Month'))
                ))
            
                prediction_results.append(f"Date: {test_year}-{month:02d}, Spawn: ({location[0]:.2f}°E, {location[1]:.2f}°N), Temp: {temperature:.2f}°C")

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

    if not prediction_results:
        prediction_results.append("No typhoons predicted within the Western Pacific region (100°E to 180°E and 0°N to 40°N).")

    return fig, html.Ul([html.Li(result) for result in prediction_results])

if __name__ == '__main__':
    interrupt_received = False
    user_input_received = threading.Event()
    should_exit = threading.Event()

    def signal_handler(signum, frame):
        global interrupt_received
        interrupt_received = True
        should_exit.set()
        print("\nInterrupt received. Do you want to save the models? (y/n)")
        
        def input_thread():
            global user_input
            user_input = input().lower()
            user_input_received.set()

        input_thread = threading.Thread(target=input_thread)
        input_thread.daemon = True
        input_thread.start()

        input_thread.join(timeout=30)

        if user_input_received.is_set():
            if user_input == 'y':
                print("Saving models...")
                save_models(avg_daily_prob, monthly_probs)
            print("Exiting the program.")
        else:
            print("\nNo input received. Automatically saving the models...")
            save_models(avg_daily_prob, monthly_probs)
            print("Exiting the program.")
        
        os._exit(0)  # Force exit the program

    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(description='Typhoon Prediction')
    parser.add_argument('--start_year', type=int, default=1950, help='Start year for training data')
    parser.add_argument('--end_year', type=int, default=2022, help='End year for training data')
    parser.add_argument('--mode', choices=['train', 'predict'], default='predict', help='Mode: train or predict')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--start_epoch', type=int, default=0, help='Epoch to start or resume training from')
    args = parser.parse_args()

    try:
        if args.mode == 'train':
            print("Preparing data...")
            count_data, spawn_data = prepare_data(args.start_year, args.end_year)
            train_models(count_data, spawn_data, nb_epochs=args.epochs, start_epoch=args.start_epoch)
            print("Training completed and models saved.")
        elif args.mode == 'predict':
            if os.path.exists(COUNT_MODEL_PATH) and os.path.exists(SPAWN_MODEL_PATH) and os.path.exists(os.path.join(MODEL_DIR, 'other_data.pkl')):
                print(f"Loading models and data...")
                count_state_dict = torch.load(COUNT_MODEL_PATH, map_location=device)
                spawn_state_dict = torch.load(SPAWN_MODEL_PATH, map_location=device)
                
                with open(os.path.join(MODEL_DIR, 'other_data.pkl'), 'rb') as f:
                    other_data = pickle.load(f)
                
                avg_daily_prob = other_data['avg_daily_prob']
                monthly_probs = other_data['monthly_probs']
                count_scaler_X = other_data['count_scaler_X']
                spawn_scaler_X = other_data['spawn_scaler_X']
                spawn_scaler_y = other_data['spawn_scaler_y']
        
                print("Loaded monthly probabilities of a typhoon:")
                for month, prob in monthly_probs.items():
                    print(f"  {calendar.month_name[month]}: {prob:.4f}")
                print(f"\nLoaded overall average daily probability: {avg_daily_prob:.4f}")

                count_input_size = 6  # year, month, day, dayofyear, temperature, count_normalized
                spawn_input_size = 6  # year, month, day, dayofyear, temperature, count_normalized
                hidden_size = 64

                count_model = CountPredictor(count_input_size, hidden_size, avg_daily_prob).to(device)
                count_model.load_state_dict(count_state_dict)
                count_model.eval()

                spawn_model = SpawnPredictor(spawn_input_size, hidden_size, output_size=4).to(device)
                spawn_model.load_state_dict(spawn_state_dict)
                spawn_model.eval()

                print("Models loaded successfully.")

                print("Starting Dash server...")
                app.run_server(debug=True)
            else:
                print(f"No models or data found in {MODEL_DIR}. Please train the models first.")
                exit(1)
    except KeyboardInterrupt:
        pass  # The signal_handler will take care of this
    except Exception as e:
        print(f"An error occurred: {e}")
    
    if interrupt_received:
        while not user_input_received.is_set():
            time.sleep(0.1)
