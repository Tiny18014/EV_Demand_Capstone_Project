"""
Dashboard Utilities
===================
REFACTOR v4: Implements a custom DashboardAgent class to encapsulate the LLM persona 
for executive-level EV market intelligence.
"""

import pandas as pd
import numpy as np
import joblib
import pickle
from pathlib import Path
import warnings
from textwrap import dedent
import plotly.express as px
import streamlit as st
import sys
import os
from openai import OpenAI

# --- Path and Model Setup ---

ROOT_DIR = Path(__file__).parent.parent.resolve()
DATA_PATH = ROOT_DIR /"data" / "cmldata" / "EV_Dataset.csv"
MODELS_DIR = ROOT_DIR /"model"
CLASSICAL_MODEL_PREFIX = "advanced_model_"

sys.path.append(str(ROOT_DIR /"model"))

# CRITICAL IMPORTS from training script
try:
    from src.model.advanced_model_trainer import create_advanced_features, prepare_data_for_training
except ImportError as e:
    print("Warning: advanced_model_trainer not found. Some features may not work.")
    create_advanced_features = None
    prepare_data_for_training = None

warnings.filterwarnings("ignore")

# --- AGENT CONFIGURATION & INITIALIZATION ---

from huggingface_hub import InferenceClient
from textwrap import dedent

# --- AGENT CONFIGURATION & INITIALIZATION ---
HF_TOKEN = st.secrets.get("DF_AGENT")
LLM_MODEL_ID = "openai/gpt-oss-20b"
LLM_CLIENT = None

if HF_TOKEN:
    try:
        LLM_CLIENT = InferenceClient(model=LLM_MODEL_ID, token=HF_TOKEN)
        print(f"Hugging Face InferenceClient initialized for {LLM_MODEL_ID}.")
    except Exception as e:
        print(f"Error initializing Hugging Face InferenceClient: {e}")
        LLM_CLIENT = None


# --- DASHBOARD AGENT CLASS ---
class DashboardAgent:
    """LLM agent generating executive insights from forecast data."""

    def __init__(self):
        self.description = dedent("""
            You are a **Tech→Business Bridge Agent** for EV demand intelligence. 
            Your role is to translate numerical forecasts into **executive-style insights**.
            Focus on market demand, growth regions, and strategic direction.
            Avoid statistical jargon — write analytically and confidently.
        """)

        self.instructions = dedent("""
            PROCESS:
            - Review key metrics: Total Sales, Top 5 States, Top 5 Categories.
            - Identify concentration of growth and emerging opportunities.
            
            OUTPUT STYLE:
            - Markdown format
            - Headline: ### Classical Model Forecast Analysis (2025)
            - 3–4 concise analytical bullets on trends and implications.
            - Tone: Executive, realistic, commercially relevant.
        """)

    def invoke(self, model_type: str, data_summary: str):
        """Builds prompt and queries the Hugging Face text-generation API."""
        if not LLM_CLIENT:
            raise RuntimeError("LLM client not initialized. Check your DF_AGENT secret or Hugging Face connection.")

        system_prompt = self.description
        user_prompt = dedent(f"""
            {self.instructions}

            INPUT DATA (Model: {model_type}):
            {data_summary}

            Now, generate the report following the OUTPUT STYLE.
        """)

        try:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            response = LLM_CLIENT.text_generation(
                prompt=full_prompt,
                max_new_tokens=512,
                temperature=0.25,
                stream=False,
            )

            if isinstance(response, str):
                return response.strip()

            if isinstance(response, dict) and "generated_text" in response:
                return response["generated_text"].strip()

            return f"⚠️ Unexpected response type: {type(response)}"

        except Exception as e:
            return f"### {model_type} Model Forecast Analysis (2025)\n\n⚠️ Agent Error: {e}"



# Instantiate global agent
report_agent = DashboardAgent()

# --- CORE FUNCTIONS ---

@st.cache_data
def get_2025_data():
    """Load and prepare data for the year 2025."""
    df = pd.read_csv("src/data/cmldata/EV_Dataset.csv", parse_dates=["Date"], low_memory=False)
    df_2025 = df[df["Date"].dt.year == 2025].copy()

    if df_2025.empty:
        df_2024 = df[df["Date"].dt.year == 2024].copy()
        if df_2024.empty:
            raise ValueError("No data for 2024 found to create synthetic 2025 dataset.")
        df_2025 = df_2024.copy()
        df_2025["Date"] = df_2025["Date"] + pd.DateOffset(years=1)
        df_2025["EV_Sales_Quantity"] = 0

    if "Vehicle_Category" not in df_2025.columns and "Vehicle_Class" in df_2025.columns:
        df_2025.rename(columns={"Vehicle_Class": "Vehicle_Category"}, inplace=True)

    df_2025["Vehicle_Category"] = df_2025["Vehicle_Category"].fillna("Unknown")
    return df_2025


@st.cache_data
def run_classical_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Run category-level classical forecast models."""
    all_preds = []
    df["Vehicle_Category"] = df["Vehicle_Category"].fillna("Unknown")
    categories = df["Vehicle_Category"].unique()
    df_features = create_advanced_features(df.copy())

    for category in categories:
        df_cat = df_features[df_features["Vehicle_Category"] == category].copy()
        if df_cat.empty:
            continue

        category_file = category.replace(" ", "_").replace("/", "_")
        model_path = MODELS_DIR / f"{CLASSICAL_MODEL_PREFIX}{category_file}.pkl"

        if not model_path.exists():
            print(f"Warning: Missing model for '{category}' → {model_path}")
            continue

        try:
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)

            model = model_data["primary_model"]
            scaler = model_data["scaler"]
            feature_names = model_data["feature_names"]

            X_scaled, _, _, _ = prepare_data_for_training(df_cat.copy(), feature_subset=feature_names)
            preds = np.maximum(0, model.predict(X_scaled)).astype(int)
            df_cat["Predicted_Sales"] = preds
            all_preds.append(df_cat)
        except Exception as e:
            print(f"Error predicting for category '{category}': {e}")

    if not all_preds:
        return pd.DataFrame()

    final_df = pd.concat(all_preds, ignore_index=True)
    return final_df[["Date", "State", "Vehicle_Category", "Predicted_Sales"]]


def generate_agent_report(predictions_df: pd.DataFrame, model_type: str) -> str:
    """Generate analytical summary using the DashboardAgent."""
    global report_agent

    if predictions_df.empty:
        return f"### {model_type} Model Forecast Analysis (2025)\n\n**Warning:** No data available."

    if not LLM_CLIENT:
        return _generate_fallback_insights_text(
            predictions_df,
            model_type,
            error_msg="LLM Agent not initialized. Check your 'DF_AGENT' secret.",
        )

    total_sales = predictions_df["Predicted_Sales"].sum()
    sales_by_state = predictions_df.groupby("State")["Predicted_Sales"].sum().sort_values(ascending=False)
    sales_by_category = predictions_df.groupby("Vehicle_Category")["Predicted_Sales"].sum().sort_values(ascending=False)

    data_summary = dedent(f"""
        Total Forecasted Sales: {total_sales:,}
        Top 5 States: {sales_by_state.head(5).to_dict()}
        Top 5 Vehicle Categories: {sales_by_category.head(5).to_dict()}
    """)

    try:
        response = report_agent.invoke(model_type, data_summary)
        
        report_text = dedent(f"""
            ### {model_type} Model Forecast Analysis (2025) 🤖
            {response}  # ✅ Use the returned string directly here
        """)
        return report_text.strip()
    except Exception as e:
        return _generate_fallback_insights_text(predictions_df, model_type, f"LLM API Error: {e}")


def _generate_fallback_insights_text(predictions_df: pd.DataFrame, model_type: str, error_msg: str):
    """Fallback report if LLM fails."""
    total_sales = predictions_df["Predicted_Sales"].sum()
    top_states = predictions_df.groupby("State")["Predicted_Sales"].sum().nlargest(3).index.tolist()
    top_categories = predictions_df.groupby("Vehicle_Category")["Predicted_Sales"].sum().nlargest(2).index.tolist()

    return dedent(f"""
        ### Generic {model_type} Forecast Analysis (2025) ⚠️
        **Agent Error:** {error_msg}

        - **Total Projected Sales:** {total_sales:,} units
        - **Top Markets:** {', '.join(top_states)}
        - **Leading Categories:** {', '.join(top_categories)}
    """).strip()


@st.cache_data
def generate_on_demand_forecast(category: str, state: str, days_to_forecast: int):
    """Generate a forecast for a specific region and vehicle type."""
    category_file = category.replace(" ", "_").replace("/", "_")
    model_path = MODELS_DIR / f"advanced_model_{category_file}.pkl"

    if not model_path.exists():
        return None, f"Error: No model found for '{category}' at {model_path}"

    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
    except Exception as e:
        return None, f"Error loading model: {e}"

    model = model_data["primary_model"]
    scaler = model_data["scaler"]
    feature_names = model_data["feature_names"]

    hist_df = pd.read_csv(DATA_PATH, parse_dates=["Date"], low_memory=False)
    if "Vehicle_Class" in hist_df.columns:
        hist_df.rename(columns={"Vehicle_Class": "Vehicle_Category"}, inplace=True)

    hist_df = hist_df[(hist_df["State"] == state) & (hist_df["Vehicle_Category"] == category)].copy()
    if hist_df.empty:
        return None, f"No historical data for '{category}' in '{state}'."

    last_date = hist_df["Date"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_forecast)
    future_df = pd.DataFrame({
        "Date": future_dates,
        "State": state,
        "Vehicle_Category": category,
        "EV_Sales_Quantity": 0
    })

    combined = pd.concat([hist_df, future_df], ignore_index=True)
    df_feat = create_advanced_features(combined)
    future_features = df_feat.iloc[-days_to_forecast:].copy()

    X_scaled, _, _, _ = prepare_data_for_training(future_features.copy(), feature_subset=feature_names)
    preds = np.maximum(0, model.predict(X_scaled)).round(0).astype(int)
    future_df["Forecasted_Sales"] = preds

    fig = px.line(future_df, x="Date", y="Forecasted_Sales",
                  title=f"{days_to_forecast}-Day Forecast for {category} in {state}")
    fig.update_traces(mode="lines+markers")
    fig.update_layout(xaxis_title="Date", yaxis_title="Forecasted Sales")

    return future_df[["Date", "Forecasted_Sales"]], fig
