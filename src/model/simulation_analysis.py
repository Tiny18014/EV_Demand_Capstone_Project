import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import sqlite3
from pathlib import Path
import pickle

def get_simulation_data(db_path):
    """Fetches actual and predicted data from the simulation database."""
    if not Path(db_path).exists():
        return pd.DataFrame()
    
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM live_predictions", conn, parse_dates=['timestamp'])
    return df

def calculate_metrics(df):
    """Calculates regression metrics from the simulation data."""
    if df.empty or 'actual_sales' not in df.columns or 'predicted_sales' not in df.columns:
        return None
    
    # Ensure there are no NaNs which can crash metric functions
    df.dropna(subset=['actual_sales', 'predicted_sales'], inplace=True)
    if df.empty:
        return None

    metrics = {
        "r2": r2_score(df['actual_sales'], df['predicted_sales']),
        "mae": mean_absolute_error(df['actual_sales'], df['predicted_sales']),
        "rmse": np.sqrt(mean_squared_error(df['actual_sales'], df['predicted_sales']))
    }
    return metrics

def create_performance_graph(df):
    """Creates a Plotly graph comparing actual vs. predicted sales over time."""
    if df.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['actual_sales'], 
        mode='lines', 
        name='Actual Sales',
        line=dict(color='#48CAE4')
    ))
    fig.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['predicted_sales'], 
        mode='lines', 
        name='Predicted Sales',
        line=dict(color='#FFD166', dash='dash')
    ))
    
    fig.update_layout(
        title="Live Simulation: Actual vs. Predicted Sales",
        xaxis_title="Timestamp",
        yaxis_title="EV Sales Quantity",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        paper_bgcolor="#112235",
        plot_bgcolor="#1c2a38",
        font=dict(color="#E0E6ED")
    )
    return fig

def get_training_performance_summary():
    """
    Finds all category-specific models, loads their saved test scores,
    and returns a summary DataFrame.
    """
    ROOT_DIR = Path(__file__).parent.resolve()  # Resolves to src/model/
    MODELS_DIR = ROOT_DIR
    
    model_files = list(MODELS_DIR.glob("advanced_model_*.pkl"))

    if not model_files:
        return pd.DataFrame() # Return empty dataframe if no models found

    performance_data = []

    for model_path in model_files:
        try:
            category_name = model_path.stem.replace("advanced_model_", "").replace("_", " ")

            # Add a check to skip the 'Unknown' category
            if category_name.lower() == 'unknown':
                continue

            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Navigate the nested dictionary to get the scores
            scores = model_data.get('test_scores', {}).get('optimized', {})
            mae = scores.get('MAE')
            r2 = scores.get('R2')

            if mae is not None and r2 is not None:
                performance_data.append({
                    "Vehicle Category": category_name,
                    "MAE": mae,
                    "R² Score": r2
                })
        except Exception:
            # Silently ignore models that can't be read or don't have scores
            continue

    if not performance_data:
        return pd.DataFrame()

    performance_df = pd.DataFrame(performance_data)
    performance_df = performance_df.sort_values(by="R² Score", ascending=False).reset_index(drop=True)
    
    return performance_df
