#!/usr/bin/env python3
"""
Advanced EV Demand Forecasting Model Trainer
Creates a high-performance Monthly LightGBM model with Daily Pattern Distribution.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
from pathlib import Path
import warnings
import optuna
import holidays

warnings.filterwarnings('ignore')

# --- PATH CONFIGURATION ---
# Assumes file is in: ProjectRoot/src/model/advanced_model_trainer.py
MODEL_DIR_PATH = Path(__file__).parent.resolve()      # src/model
SRC_DIR = MODEL_DIR_PATH.parent.resolve()             # src
ROOT_DIR = SRC_DIR.parent.resolve()                   # ProjectRoot

# Consistent Paths
DATA_PATH = SRC_DIR / "data" / "cmldata" / "EV_Dataset.csv"
MODELS_DIR = MODEL_DIR_PATH                           # Save models in src/model

print("🚀 Advanced EV Demand Forecasting Model Trainer (Monthly + Daily Hybrid)")
print(f"📂 Data Path: {DATA_PATH}")
print(f"📂 Model Path: {MODELS_DIR}")
print("=" * 60)

def load_and_clean_data():
    """Load data and standardize categories."""
    print("📉 Loading data...")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"❌ Data file not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    
    if 'Vehicle_Category' not in df.columns and 'Vehicle_Class' in df.columns:
        df.rename(columns={'Vehicle_Class': 'Vehicle_Category'}, inplace=True)
    
    df['Vehicle_Category'] = df['Vehicle_Category'].fillna('Unknown')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Standardize categories just in case
    category_map = {
        'TWO WHEELER(NT)': '2-Wheelers', 'TWO WHEELER (INVALID CARRIAGE)': '2-Wheelers',
        'M-CYCLE/SCOOTER': '2-Wheelers', 'MOTOR CYCLE/SCOOTER-USED FOR HIRE': '2-Wheelers',
        'THREE WHEELER(T)': '3-Wheelers', 'THREE WHEELER(NT)': '3-Wheelers',
        'MOTOR CAR': '4-Wheelers', 'MOTOR CAB': '4-Wheelers', 'LIGHT MOTOR VEHICLE': '4-Wheelers',
        'BUS': 'Bus', 'HEAVY PASSENGER VEHICLE': 'Bus', 'MEDIUM PASSENGER VEHICLE': 'Bus'
    }
    # Apply mapping only to values not in standard set
    standard_cats = ['2-Wheelers', '3-Wheelers', '4-Wheelers', 'Bus', 'Others']
    df['Vehicle_Category'] = df['Vehicle_Category'].apply(lambda x: category_map.get(x, x))
    df.loc[~df['Vehicle_Category'].isin(standard_cats), 'Vehicle_Category'] = 'Others'
    
    return df

def extract_daily_patterns(df):
    """
    Extracts the daily distribution weights from historical daily data (2021-2024).
    Returns a dictionary: {Category: {Month: {Day: weight}}}
    """
    print("📅 Extracting daily patterns from historical data (2021-2024)...")
    
    df_hist = df[df['Date'].dt.year <= 2024].copy()
    df_hist['Month'] = df_hist['Date'].dt.month
    df_hist['Day'] = df_hist['Date'].dt.day
    
    daily_sales = df_hist.groupby(['Vehicle_Category', 'Month', 'Day'])['EV_Sales_Quantity'].sum().reset_index()
    
    monthly_sales = daily_sales.groupby(['Vehicle_Category', 'Month'])['EV_Sales_Quantity'].transform('sum')
    daily_sales['Weight'] = daily_sales['EV_Sales_Quantity'] / monthly_sales
    daily_sales['Weight'] = daily_sales['Weight'].fillna(1.0 / 30.0) 
    
    patterns = {}
    for cat in df_hist['Vehicle_Category'].unique():
        patterns[cat] = {}
        cat_data = daily_sales[daily_sales['Vehicle_Category'] == cat]
        for month in range(1, 13):
            month_data = cat_data[cat_data['Month'] == month]
            day_weights = dict(zip(month_data['Day'], month_data['Weight']))
            patterns[cat][month] = day_weights

    print(f"✅ Extracted patterns for {len(patterns)} categories.")
    return patterns

def aggregate_to_monthly(df):
    """Aggregates daily data to monthly level for robust forecasting."""
    print("📦 Aggregating data to Monthly level...")
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    
    monthly_df = df.groupby(['State', 'Vehicle_Category', 'Year', 'Month'])['EV_Sales_Quantity'].sum().reset_index()
    monthly_df['Date'] = pd.to_datetime(monthly_df[['Year', 'Month']].assign(Day=1))
    
    print(f"✅ Aggregated to {len(monthly_df)} monthly records.")
    return monthly_df

def create_monthly_features(df):
    """Create features for the monthly model."""
    df = df.sort_values(['State', 'Vehicle_Category', 'Date']).copy()
    
    # 1. Temporal Features
    df['Quarter'] = df['Date'].dt.quarter
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month']/12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month']/12)
    
    # 2. Lag Features
    for lag in [1, 2, 3, 6, 12]:
        df[f'Lag_{lag}'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].shift(lag)

    # 3. Rolling Features
    for window in [3, 6, 12]:
        df[f'Roll_Mean_{window}'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f'Roll_Std_{window}'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
        )

    # 4. EWMA
    for span in [3, 12]:
        df[f'EWMA_{span}'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].transform(
            lambda x: x.shift(1).ewm(span=span).mean()
        )

    df = df.fillna(0)
    return df

def train_monthly_model(df_train, category):
    """Trains a LightGBM model for a specific category on monthly data."""
    
    features = [c for c in df_train.columns if c not in ['Date', 'EV_Sales_Quantity', 'State', 'Vehicle_Category', 'Year']]
    
    df_train['State'] = df_train['State'].astype('category')
    state_codes = df_train['State'].cat.codes
    features.append('State_Code')
    df_train['State_Code'] = state_codes
    
    X = df_train[features]
    y = df_train['EV_Sales_Quantity']
    
    if len(X) < 50:
        return None, None, None, float('inf')

    # Train/Val Split (Cutoff mid-2024)
    cutoff_date = pd.Timestamp('2024-06-01')
    mask_train = df_train['Date'] < cutoff_date
    mask_val = df_train['Date'] >= cutoff_date
    
    X_train, y_train = X[mask_train], y[mask_train]
    X_val, y_val = X[mask_val], y[mask_val]
    
    if len(X_val) == 0: # Fallback if no validation data
        X_train, y_train = X[:-6], y[:-6]
        X_val, y_val = X[-6:], y[-6:]

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'n_estimators': 2000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 64),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'verbosity': -1,
            'n_jobs': -1
        }
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        
        return mean_absolute_error(y_val, model.predict(X_val_scaled))
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10) # Reduced to 10 for speed

    best_params = study.best_params
    best_params.update({'objective': 'regression', 'metric': 'mae', 'n_estimators': 2000, 'n_jobs': -1, 'verbosity': -1})
    
    final_model = lgb.LGBMRegressor(**best_params)
    
    X_full = scaler.fit_transform(X)
    final_model.fit(X_full, y)

    y_pred = final_model.predict(X_full)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    print(f"  --> {category} Model: R2={r2:.4f}, MAE={mae:.2f}")

    return final_model, scaler, features, mae

def main_train():
    try:
        df = load_and_clean_data()
    except FileNotFoundError as e:
        print(e)
        return

    daily_patterns = extract_daily_patterns(df)
    df_monthly = aggregate_to_monthly(df)
    df_features = create_monthly_features(df_monthly)

    model_path = MODELS_DIR / "advanced_model_monthly_hybrid.pkl"
    existing_models = {}
    if model_path.exists():
        try:
            with open(model_path, 'rb') as f:
                existing_models = pickle.load(f)
            print(f"ℹ️  Found existing model file with {len(existing_models)} categories.")
        except Exception:
            print("⚠️ Could not load existing models. Starting fresh.")

    new_models_bundle = existing_models.copy()
    categories = df_features['Vehicle_Category'].unique()

    updates_made = False

    for cat in categories:
        print(f"\n🚗 Training Monthly Model for: {cat}")
        cat_df = df_features[df_features['Vehicle_Category'] == cat].copy()

        model, scaler, feature_names, new_mae = train_monthly_model(cat_df, cat)
        
        if model:
            should_update = True
            if cat in existing_models and 'mae' in existing_models[cat]:
                old_mae = existing_models[cat]['mae']
                print(f"  🔍 Comparison: New MAE ({new_mae:.2f}) vs Old MAE ({old_mae:.2f})")
                if new_mae > old_mae:
                    print(f"  ❌ Performance degraded. Keeping old model.")
                    should_update = False
                else:
                    print(f"  ✅ Performance improved or matched. Updating model.")

            if should_update:
                new_models_bundle[cat] = {
                    'model': model,
                    'scaler': scaler,
                    'features': feature_names,
                    'daily_patterns': daily_patterns.get(cat, {}),
                    'states': cat_df['State'].unique().tolist(),
                    'mae': new_mae
                }
                updates_made = True

    if updates_made or not model_path.exists():
        with open(model_path, 'wb') as f:
            pickle.dump(new_models_bundle, f)
        print(f"\n✅ Saved updated hybrid models to {model_path}")
    else:
        print("\n⏹️  No performance improvements found. Existing models retained.")

def predict_daily_2026():
    """Generates daily predictions for 2026 using the hybrid model."""
    print("\n🔮 Generating Daily Forecasts for 2026...")
    model_path = MODELS_DIR / "advanced_model_monthly_hybrid.pkl"
    if not model_path.exists():
        print("❌ Model not found. Run training first.")
        return None
        
    with open(model_path, 'rb') as f:
        models_data = pickle.load(f)
        
    df = load_and_clean_data()
    df_monthly = aggregate_to_monthly(df)
    df_features = create_monthly_features(df_monthly)

    all_predictions = []
    future_dates = pd.date_range(start='2026-01-01', end='2026-12-31', freq='MS')

    for cat, model_data in models_data.items():
        model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['features']
        patterns = model_data['daily_patterns']
        states = model_data['states']

        cat_history = df_features[df_features['Vehicle_Category'] == cat].copy()

        for state in states:
            state_history = cat_history[cat_history['State'] == state].sort_values('Date')
            if state_history.empty: continue

            current_history = state_history.copy()

            for date in future_dates:
                # Create a row for this future date
                new_row = pd.DataFrame([{
                    'State': state, 'Vehicle_Category': cat, 'Date': date,
                    'Year': date.year, 'Month': date.month, 'EV_Sales_Quantity': 0
                }])
                temp_df = pd.concat([current_history, new_row], ignore_index=True)
                temp_features = create_monthly_features(temp_df)
                
                row_to_predict = temp_features.iloc[[-1]].copy()
                row_to_predict['State_Code'] = pd.Categorical(row_to_predict['State'], categories=pd.Categorical(states).categories).codes

                X_pred = row_to_predict[feature_names]
                X_pred_scaled = scaler.transform(X_pred)

                pred_monthly_total = max(0, model.predict(X_pred_scaled)[0])

                current_history = pd.concat([current_history, new_row], ignore_index=True)
                current_history.iloc[-1, current_history.columns.get_loc('EV_Sales_Quantity')] = pred_monthly_total

                # Distribute to Daily
                days_in_month = pd.Period(date, freq='M').days_in_month
                month_weights = patterns.get(date.month, {})

                for day in range(1, days_in_month + 1):
                    weight = month_weights.get(day, 1.0/days_in_month)
                    daily_sale = pred_monthly_total * weight
                    all_predictions.append({
                        'Date': pd.Timestamp(year=2026, month=date.month, day=day),
                        'State': state,
                        'Vehicle_Category': cat,
                        'Predicted_Sales': int(daily_sale)
                    })

    pred_df = pd.DataFrame(all_predictions)
    output_path = ROOT_DIR / "output" / "daily_predictions_2026.csv"
    output_path.parent.mkdir(exist_ok=True)
    pred_df.to_csv(output_path, index=False)
    print(f"✅ Generated {len(pred_df)} daily predictions for 2026.")
    return output_path

# =================================================================================================
# LEGACY COMPATIBILITY FUNCTIONS
# Restored to support older scripts (dashboard_utils.py) that might import these
# =================================================================================================

def create_advanced_features(df):
    """Legacy: Create feature set for DAILY predictions."""
    df = df.copy()
    if 'Vehicle_Category' in df.columns:
        df['Vehicle_Category'] = df['Vehicle_Category'].fillna('Unknown')

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['State', 'Vehicle_Category', 'Date'])

    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)

    for lag in [1, 7, 30]:
        df[f'lag_{lag}'] = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].shift(lag)

    for window in [7, 30]:
        grouped_rolling = df.groupby(['State', 'Vehicle_Category'])['EV_Sales_Quantity'].rolling(window=window, min_periods=1)
        df[f'rolling_mean_{window}'] = grouped_rolling.mean().reset_index(level=[0, 1], drop=True)
        df[f'rolling_std_{window}'] = grouped_rolling.std().reset_index(level=[0, 1], drop=True)

    df = df.fillna(0)
    return df

def prepare_features_for_prediction(df, feature_names, scaler):
    """Legacy: Prepares a dataframe for prediction using a pre-fitted scaler."""
    if 'Month_Name' in df.columns: df = df.drop(columns=['Month_Name'])
    
    # Ensure categorical columns
    for col in ['State', 'Vehicle_Category']:
        if col in df.columns: df[col] = df[col].astype('category')
        if col in df.columns: df[col] = df[col].cat.codes

    # Select and Transform
    feature_columns = [f for f in feature_names if f in df.columns]
    X = df[feature_columns]
    return scaler.transform(X)

def prepare_data_for_training(df, target_col='EV_Sales_Quantity', feature_subset=None):
    """
    Legacy: Prepares data for training (Splitting X, y and Scaling).
    Used by older dashboard_utils.py logic.
    """
    df = df.copy()
    
    # Simple encoding for legacy support
    if 'State' in df.columns: df['State'] = df['State'].astype('category').cat.codes
    if 'Vehicle_Category' in df.columns: df['Vehicle_Category'] = df['Vehicle_Category'].astype('category').cat.codes
    
    drop_cols = ['Date', target_col, 'Month_Name']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    if feature_subset:
        feature_cols = [f for f in feature_subset if f in feature_cols]
        
    X = df[feature_cols]
    y = df[target_col]
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, feature_cols

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'predict':
        predict_daily_2026()
    else:
        main_train()