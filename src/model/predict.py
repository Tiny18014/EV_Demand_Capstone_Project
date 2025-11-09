import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import pennylane as qml
from sklearn.model_selection import train_test_split
import joblib

# --- Vehicle class mapping function ---
def map_vehicle_category(vehicle) -> str:
    if pd.isna(vehicle):
        return "unknown"
    v = str(vehicle).upper()
    # 2-WHEELERS
    if any(x in v for x in ["M-CYCLE", "MOTOR CYCLE", "MOTORISED CYCLE", "SCOOTER", "MOPED"]):
        return "2-wheelers"
    # 3-WHEELERS
    if "THREE WHEELER" in v or "E-RICKSHAW" in v or "QUADRICYCLE" in v:
        return "3-wheelers"
    # 4-WHEELERS
    if any(x in v for x in ["MOTOR CAR", "MOTOR CAB", "MAXI CAB", "LUXURY CAB", "CASH VAN",
                            "PRIVATE SERVICE VEHICLE", "OMNI BUS (PRIVATE USE)",
                            "MOTOR CARAVAN", "CAMPER VAN", "HEARSES"]):
        return "4-wheelers"
    # BUS
    if "BUS" in v:
        return "bus"
    # EVERYTHING ELSE
    return "other"

# --- Comprehensive list of vehicle categories ---
vehicle_categories = [
    "ADAPTED VEHICLE",
    "AGRICULTURAL TRACTOR",
    "ARTICULATED VEHICLE",
    "AUXILIARY TRAILER",
    "BUS",
    "CONSTRUCTION EQUIPMENT VEHICLE",
    "CONSTRUCTION EQUIPMENT VEHICLE (COMMERCIAL)",
    "EDUCATIONAL INSTITUTION BUS",
    "E-RICKSHAW(P)",
    "E-RICKSHAW WITH CART (G)",
    "FORK LIFT",
    "GOODS CARRIER",
    "LUXURY CAB",
    "MAXI CAB",
    "M-CYCLE/SCOOTER",
    "M-CYCLE/SCOOTER-WITH SIDE CAR",
    "MOBILE CLINIC",
    "MOPED",
    "MOTOR CAB",
    "MOTOR CAR",
    "MOTOR CYCLE/SCOOTER-SIDECAR(T)",
    "MOTOR CYCLE/SCOOTER-USED FOR HIRE",
    "MOTORISED CYCLE (CC > 25CC)",
    "OMNI BUS",
    "OMNI BUS (PRIVATE USE)",
    "PRIVATE SERVICE VEHICLE",
    "PRIVATE SERVICE VEHICLE (INDIVIDUAL USE)",
    "SEMI-TRAILER (COMMERCIAL)",
    "THREE WHEELER (GOODS)",
    "THREE WHEELER (PASSENGER)",
    "THREE WHEELER (PERSONAL)",
    "TRACTOR (COMMERCIAL)"
]

# --- State group list ---
state_ev_groups = [
    "high_ev_states", "moderate_ev_states", "low_ev_states"
]  # replace if you have more granular categories

months = [1,2,3,4,5,6,7,8,9,10, 11, 12]  # October, November, December
year = 2025
start_ref_date = pd.to_datetime("2014-01-01")

# --- Generate all combinations ---
rows = []
for month, category, state in product(months, vehicle_categories, state_ev_groups):
    date = pd.Timestamp(year, month, 1)
    rows.append({
        "Date": date,
        "Year": year,
        "Month": month,
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "Days_Since_Start": (date - start_ref_date).days,
        "Vehicle_Class": category,
        "State_EV_Group": state
    })

future_df = pd.DataFrame(rows)

# --- Apply your vehicle class mapping ---
future_df["Vehicle_Category"] = future_df["Vehicle_Class"].apply(map_vehicle_category)

# --- Add placeholder for target ---
future_df["Log_EV_Sales_Quantity"] = np.nan

# --- Reorder for clarity ---
future_df = future_df[
    [
        "Date", "Year", "Month", "month_sin", "month_cos",
        "Days_Since_Start", "Vehicle_Category", "Vehicle_Class",
        "State_EV_Group", "Log_EV_Sales_Quantity"
    ]
]

# --- Preview summary ---
print(f"Generated {len(future_df):,} rows ({len(vehicle_categories)} vehicle types × {len(state_ev_groups)} states × {len(months)} months)")
categorical_cols = ['Vehicle_Class', 'Vehicle_Category']
df = future_df.copy()
for col in categorical_cols:
    e = LabelEncoder()
    df[col] = e.fit_transform(df[col])
group_le = LabelEncoder()
df['State_EV_Group'] = group_le.fit_transform(df['State_EV_Group'])
scaler = MinMaxScaler(feature_range=(-1, 1))
features_to_use = ['Vehicle_Class', 'State_EV_Group', 'Year',
                      'Vehicle_Category','month_sin', 'month_cos']
X = scaler.fit_transform(df[features_to_use])
df['Date'] = pd.to_datetime(df['Date'])
df["Days_Since_Start"] = (df["Date"] - pd.to_datetime("2014-01-01")).dt.days
print(df["Days_Since_Start"])
scaler = StandardScaler()
df[['Year', 'Month_scaled', 'month_sin', 'month_cos', 'Days_Since_Start']] = scaler.fit_transform(df[['Year', 'Month', 'month_sin', 'month_cos', 'Days_Since_Start']])
finalcols = ['Vehicle_Class', 'Vehicle_Category','Month', 'month_sin', 'month_cos', 'State_EV_Group', 'Days_Since_Start','Log_EV_Sales_Quantity' ]
dff = df[finalcols]
print(df[["Month_scaled", "Month","Days_Since_Start" ]].drop_duplicates())
"""
X = dff[['Vehicle_Class', 'Vehicle_Category', 'Month', 'month_sin', 'month_cos', 'State_EV_Group', 'Days_Since_Start']].values


dev = qml.device('lightning.qubit', wires=len(X[0]))  # One wire per feature

@qml.qnode(dev)
def quantum_embedding(x):
    for i in range(len(x)):
        qml.RX(x[i], wires=i)

    qml.templates.AngleEmbedding(x, wires=range(len(x)))
    for i in range(len(x)):
        qml.RY(np.pi * x[i], wires=i)
    return [qml.expval(qml.PauliZ(i)) for i in range(len(x))]

def generate_embeddings(X):
    embeddings = []
    for row in X:
        embeddings.append(quantum_embedding(row))  # Run quantum circuit for each row
    return np.array(embeddings)

X_future_embeddings = generate_embeddings(X)
model3 = joblib.load("src/model/xgboostqml.joblib")
y_pred = model3.predict(X_future_embeddings)
print(y_pred)"""

