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
    Returns the training performance summary.
    Prioritizes loading the 'final_model_accuracy_report.csv' if available,
    as it contains the scientifically verified metrics.
    Falls back to inspecting pickle files if the report is missing.
    """
    # 1. Try loading the CSV report first (Most Accurate)
    # Adjust path to where report_model_accuracy.py saves it
    # Assuming this script runs from src/model/, we go up to project root then output/
    CSV_PATH = Path(__file__).parent.parent.parent.resolve() / "output" / "final_model_accuracy_report.csv"
    
    if CSV_PATH.exists():
        try:
            df = pd.read_csv(CSV_PATH)
            # Map CSV columns to Dashboard columns
            # CSV: Category, Model_Type, Test_Records, MAE, RMSE, R2_Score
            # Dashboard Expects: "Vehicle Category", "MAE", "R² Score"
            
            summary = df.rename(columns={
                'Category': 'Vehicle Category',
                'R2_Score': 'R² Score'
            })
            return summary[['Vehicle Category', 'MAE', 'R² Score']].sort_values('R² Score', ascending=False)
        except Exception:
            pass # Fallback to old logic if CSV read fails

    # 2. Fallback: Look for .pkl files (Old Logic + Specialized Support)
    ROOT_DIR = Path(__file__).parent.resolve()
    
    # Find standard models
    standard_models = list(ROOT_DIR.glob("advanced_model_*.pkl"))
    # Find specialized models
    specialized_models = list(ROOT_DIR.glob("specialized_*_model.pkl"))
    
    all_models = standard_models + specialized_models

    if not all_models:
        return pd.DataFrame()

    performance_data = []

    for model_path in all_models:
        try:
            # Determine Category Name
            name = model_path.stem
            if "specialized" in name:
                # format: specialized_bus_monthly_model
                category_name = name.split("_")[1].capitalize() # e.g., 'Bus'
                if category_name == '3w': category_name = '3-Wheelers'
            else:
                # format: advanced_model_2-Wheelers
                category_name = name.replace("advanced_model_", "").replace("_", " ")

            if category_name.lower() == 'unknown': continue

            # Load metrics
            # Note: Specialized models might not have 'test_scores' dict inside if they are just the model object.
            # This fallback loop is brittle for specialized models unless they were saved with metadata.
            # The CSV method above is much safer.
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Handle Dictionary format (Standard Models)
            if isinstance(model_data, dict) and 'test_scores' in model_data:
                scores = model_data['test_scores'].get('optimized', {})
                mae = scores.get('MAE')
                r2 = scores.get('R2')
                
                if mae is not None:
                    performance_data.append({
                        "Vehicle Category": category_name,
                        "MAE": mae,
                        "R² Score": r2 if r2 else 0.0
                    })
            
            # Handle Specialized Models (If they don't have metadata, we skip or mock)
            # Since we generated a CSV report, we rely on that primarily.
            
        except Exception:
            continue

    if not performance_data:
        return pd.DataFrame()

    performance_df = pd.DataFrame(performance_data)
    return performance_df.sort_values(by="R² Score", ascending=False).reset_index(drop=True)