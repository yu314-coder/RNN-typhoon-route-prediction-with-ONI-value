import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
import cachetools
import functools
import tropycal.tracks as tracks
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import joblib
import calendar
import argparse

# Load IBTrACS data
ibtracs = tracks.TrackDataset(basin='west_pacific', source='ibtracs')

# Define the path for ONI data and model
ONI_DATA_PATH = os.path.join(os.getcwd(), 'oni_data.csv')
MODEL_PATH = os.path.join(os.getcwd(), 'hurricane_rnn_model.pth')
SCALER_PATH = os.path.join(os.getcwd(), 'scaler.joblib')

@cachetools.cached(cache={})
def fetch_oni_data_from_csv(file_path):
    df = pd.read_csv(file_path, sep=',', header=0, na_values='-99.90')
    df.columns = ['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    df = df.melt(id_vars=['Year'], var_name='Month', value_name='ONI')
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + df['Month'], format='%Y%b')
    df = df.set_index('Date')
    return df

# Load ONI data
oni_df = fetch_oni_data_from_csv(ONI_DATA_PATH)

@functools.lru_cache(maxsize=None)
def get_storm_data(storm_id):
    return ibtracs.get_storm(storm_id)

def filter_west_pacific_coordinates(lons, lats):
    mask = (100 <= lons) & (lons <= 180) & (0 <= lats) & (lats <= 40)
    return lons[mask], lats[mask]

def compute_fourier_components(lons, lats, n_components=10):
    t = np.linspace(0, 2*np.pi, len(lons))
    components = []
    for i in range(1, n_components+1):
        components.extend([
            np.sin(i*t),
            np.cos(i*t)
        ])
    return np.array(components).T

def prepare_data(start_year, end_year, sequence_length, n_fourier_components=10):
    X, y = [], []
    for year in range(start_year, end_year + 1):
        season = ibtracs.get_season(year)
        for storm_id in season.summary()['id']:
            storm = get_storm_data(storm_id)
            lons, lats = filter_west_pacific_coordinates(np.array(storm.lon), np.array(storm.lat))
            
            if len(lons) < sequence_length + 1:
                continue
            
            dates = pd.to_datetime(storm.time)
            oni_values = [oni_df.loc[date.strftime('%Y-%m')]['ONI'] for date in dates]
            
            fourier_components = compute_fourier_components(lons, lats, n_fourier_components)
            
            for i in range(len(lons) - sequence_length):
                X.append(np.column_stack((
                    dates[i:i+sequence_length].astype(int) / 10**9,  # Convert to Unix timestamp
                    np.full(sequence_length, dates[i].month),
                    oni_values[i:i+sequence_length],
                    lons[i:i+sequence_length],
                    lats[i:i+sequence_length],
                    fourier_components[i:i+sequence_length]
                )))
                y.append(np.array([lons[i+sequence_length], lats[i+sequence_length]]))
    
    return np.array(X), np.array(y)

class HurricaneRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        super(HurricaneRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.bn(out[:, -1, :])
        out = self.fc(out)
        return out

def train_model(X, y, sequence_length, nb_epochs=150, batch_size=32, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Normalize the data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    y_scaled = scaler_y.fit_transform(y)
    
    # Save scalers
    joblib.dump((scaler_X, scaler_y), SCALER_PATH)
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    y_tensor = torch.FloatTensor(y_scaled).to(device)
    
    # Create DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    input_size = X.shape[2]
    hidden_size = 256
    output_size = 2  # predicting next lon and lat
    model = HurricaneRNN(input_size, hidden_size, output_size).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)
    
    # Training loop
    for epoch in tqdm(range(nb_epochs)):
        model.train()
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_tensor), y_tensor).item()
        
        scheduler.step(val_loss)
        
        print(f"Epoch [{epoch+1}/{nb_epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    torch.save(model.state_dict(), MODEL_PATH)
    return model, scaler_X, scaler_y

def predict_future(model, scaler_X, scaler_y, input_sequence, date, oni_value):
    month = pd.to_datetime(date).month
    lons, lats = input_sequence[:, 0], input_sequence[:, 1]
    fourier_components = compute_fourier_components(lons, lats)
    
    input_seq = np.column_stack((
        np.full(len(input_sequence), date.timestamp()),
        np.full(len(input_sequence), month),
        np.full(len(input_sequence), oni_value),
        lons,
        lats,
        fourier_components
    ))
    
    input_seq_scaled = scaler_X.transform(input_seq)
    input_tensor = torch.FloatTensor(input_seq_scaled).unsqueeze(0).to(next(model.parameters()).device)
    
    with torch.no_grad():
        prediction_scaled = model(input_tensor)
    
    prediction = scaler_y.inverse_transform(prediction_scaled.cpu().numpy())
    return prediction[0]

def generate_future_route(model, scaler_X, scaler_y, start_point, oni_value, start_month, num_steps=10):
    route = [start_point]
    current_sequence = np.tile(start_point, (10, 1))
    current_date = pd.Timestamp(year=2023, month=start_month, day=1)
    
    for _ in range(num_steps):
        next_point = predict_future(model, scaler_X, scaler_y, current_sequence, current_date, oni_value)
        route.append(next_point)
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = next_point
        current_date += pd.Timedelta(days=1)
    
    return np.array(route)

def fit_fourier_series(lons, lats, n_components=10):
    t = np.linspace(0, 2*np.pi, len(lons))
    
    def fourier_series(t, *params):
        result = params[0]
        for i in range(1, len(params), 2):
            result += params[i] * np.sin((i+1)//2 * t) + params[i+1] * np.cos((i+1)//2 * t)
        return result
    
    params_lon, _ = curve_fit(fourier_series, t, lons, p0=[0]*(2*n_components+1))
    params_lat, _ = curve_fit(fourier_series, t, lats, p0=[0]*(2*n_components+1))
    
    return params_lon, params_lat

def fourier_equation(params):
    equation = f"{params[0]:.4f}"
    for i in range(1, len(params), 2):
        equation += f" + {params[i]:.4f}*sin({(i+1)//2}x) + {params[i+1]:.4f}*cos({(i+1)//2}x)"
    return equation

# Main execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Typhoon Analysis and Prediction Dashboard')
    parser.add_argument('--start_year', type=int, default=1950, help='Start year for training data')
    parser.add_argument('--end_year', type=int, default=2022, help='End year for training data')
    args = parser.parse_args()

    print("Preparing data...")
    X, y = prepare_data(args.start_year, args.end_year, sequence_length=10)
    
    print("Training model...")
    model, scaler_X, scaler_y = train_model(X, y, sequence_length=10)
    print("Model training completed.")

    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1("Typhoon Analysis and Prediction Dashboard"),
        
        html.Div([
            dcc.Input(id='test-year', type='number', placeholder='Test Year', value=2023, min=2022, max=2024, step=1),
            dcc.Dropdown(
                id='enso-dropdown',
                options=[
                    {'label': 'All Years', 'value': 'all'},
                    {'label': 'El Niño Years', 'value': 'el_nino'},
                    {'label': 'La Niña Years', 'value': 'la_nina'},
                    {'label': 'Neutral Years', 'value': 'neutral'}
                ],
                value='all'
            ),
            html.Button('Analyze and Predict', id='analyze-button', n_clicks=0),
        ]),

        dcc.Graph(id='typhoon-routes-graph'),
        html.Div(id='fourier-equations'),
        html.Div(id='nn-prediction-results'),
        html.Div(id='monthly-typhoon-count'),

        html.Div([
            html.Button('Show Clusters', id='show-clusters-button', n_clicks=0),
            html.Button('Show Typhoon Routes', id='show-routes-button', n_clicks=0),
            html.Button('Show Predictions', id='show-predictions-button', n_clicks=0),
        ]),
    ])

    @app.callback(
        [Output('typhoon-routes-graph', 'figure'),
         Output('fourier-equations', 'children'),
         Output('nn-prediction-results', 'children'),
         Output('monthly-typhoon-count', 'children')],
        [Input('analyze-button', 'n_clicks'),
         Input('show-clusters-button', 'n_clicks'),
         Input('show-routes-button', 'n_clicks'),
         Input('show-predictions-button', 'n_clicks')],
        [State('test-year', 'value'),
         State('enso-dropdown', 'value')]
    )
    def update_analysis(analyze_clicks, show_clusters_clicks, show_routes_clicks,
                        show_predictions_clicks, test_year, enso_value):
        ctx = dash.callback_context
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        fig_routes = go.Figure()

        # Predict typhoon routes for the test year
        test_season = ibtracs.get_season(test_year)
        predictions = []
        monthly_counts = {i: 0 for i in range(1, 13)}
        
        for storm_id in test_season.summary()['id']:
            storm = get_storm_data(storm_id)
            lons, lats = filter_west_pacific_coordinates(np.array(storm.lon), np.array(storm.lat))
            
            if len(lons) < 11:  # We need at least 11 points (10 for input, 1 for ground truth)
                continue
            
            storm_date = storm.time[0]
            oni_value = oni_df.loc[storm_date.strftime('%Y-%m')]['ONI']
            month = storm_date.month
            monthly_counts[month] += 1
            
            input_sequence = np.column_stack((lons[:10], lats[:10]))
            predicted_route = generate_future_route(model, scaler_X, scaler_y, input_sequence[0], oni_value, month, num_steps=20)
            
            predictions.append({
                'input': input_sequence,
                'predicted': predicted_route,
                'actual': np.column_stack((lons[10:], lats[10:]))
            })
            
            if button_id in ['analyze-button', 'show-predictions-button']:
                fig_routes.add_trace(go.Scattergeo(
                    lon=predicted_route[:, 0],
                    lat=predicted_route[:, 1],
                    mode='lines+markers',
                    name=f'Predicted Storm {storm_id}',
                    line=dict(color='blue'),
                    marker=dict(color='blue', size=5)
                ))
                fig_routes.add_trace(go.Scattergeo(
                    lon=lons,
                    lat=lats,
                    mode='lines+markers',
                    name=f'Actual Storm {storm_id}',
                    line=dict(color='red'),
                    marker=dict(color='red', size=5)
                ))

        if button_id in ['analyze-button', 'show-routes-button']:
            for storm_id in test_season.summary()['id']:
                storm = get_storm_data(storm_id)
                lons, lats = filter_west_pacific_coordinates(np.array(storm.lon), np.array(storm.lat))
                fig_routes.add_trace(go.Scattergeo(
                    lon=lons, lat=lats,
                    mode='lines',
                    name=f'Storm {storm_id}',
                    line=dict(width=1)
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

        # Compute Fourier series for each predicted route
        fourier_equations = []
        for i, pred in enumerate(predictions):
            params_lon, params_lat = fit_fourier_series(pred['predicted'][:, 0], pred['predicted'][:, 1])
            lon_eq = fourier_equation(params_lon)
            lat_eq = fourier_equation(params_lat)
            fourier_equations.append(html.Div([
                html.H4(f"Storm {i+1} Fourier Series Equations:"),
                html.P(f"Longitude: {lon_eq}"),
                html.P(f"Latitude: {lat_eq}")
            ]))

        def calculate_error(predicted, actual):
            min_length = min(len(predicted), len(actual))
            return np.mean(np.linalg.norm(predicted[:min_length] - actual[:min_length], axis=1))

        for i, p in enumerate(predictions):
            print(f"Storm {i+1}: Predicted length = {len(p['predicted'])}, Actual length = {len(p['actual'])}")

        nn_results = html.Div([
            html.H3(f"Neural Network Predictions for {test_year}"),
            html.P(f"Number of storms predicted: {len(predictions)}"),
            html.P(f"Average prediction error: {np.mean([calculate_error(p['predicted'], p['actual']) for p in predictions]):.2f} degrees")
        ])




     






        monthly_count_results = html.Div([
            html.H3(f"Monthly Typhoon Count for {test_year}"),
            html.Ul([html.Li(f"{calendar.month_name[month]}: {count}") for month, count in monthly_counts.items() if count > 0])
        ])

        return fig_routes, html.Div(fourier_equations), nn_results, monthly_count_results

    print("Starting Dash server...")
    app.run_server(debug=True)