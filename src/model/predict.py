import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import pennylane as qml
import joblib

def forecast_ev_sales(year=2025, months=[9, 10, 11, 12]):
    """
    Generate future EV sales forecasts using a hybrid quantum-classical model.
    Returns a DataFrame containing predicted log EV sales quantities
    with corresponding month, vehicle class, category, and state group.
    """

    # --- Vehicle class mapping function ---
    def map_vehicle_category(vehicle) -> str:
        if pd.isna(vehicle):
            return "unknown"
        v = str(vehicle).upper()
        if any(x in v for x in ["M-CYCLE", "MOTOR CYCLE", "MOTORISED CYCLE", "SCOOTER", "MOPED"]):
            return "2-wheelers"
        if "THREE WHEELER" in v or "E-RICKSHAW" in v or "QUADRICYCLE" in v:
            return "3-wheelers"
        if any(x in v for x in ["MOTOR CAR", "MOTOR CAB", "MAXI CAB", "LUXURY CAB", "CASH VAN",
                                "PRIVATE SERVICE VEHICLE", "OMNI BUS (PRIVATE USE)",
                                "MOTOR CARAVAN", "CAMPER VAN", "HEARSES"]):
            return "4-wheelers"
        if "BUS" in v:
            return "bus"
        return "other"

    # --- Categories and state groups ---
    vehicle_categories = [
        "ADAPTED VEHICLE", "AGRICULTURAL TRACTOR", "ARTICULATED VEHICLE", "AUXILIARY TRAILER", "BUS",
        "CONSTRUCTION EQUIPMENT VEHICLE", "CONSTRUCTION EQUIPMENT VEHICLE (COMMERCIAL)",
        "EDUCATIONAL INSTITUTION BUS", "E-RICKSHAW(P)", "E-RICKSHAW WITH CART (G)", "FORK LIFT",
        "GOODS CARRIER", "LUXURY CAB", "MAXI CAB", "M-CYCLE/SCOOTER", "M-CYCLE/SCOOTER-WITH SIDE CAR",
        "MOBILE CLINIC", "MOPED", "MOTOR CAB", "MOTOR CAR", "MOTOR CYCLE/SCOOTER-SIDECAR(T)",
        "MOTOR CYCLE/SCOOTER-USED FOR HIRE", "MOTORISED CYCLE (CC > 25CC)", "OMNI BUS", "OMNI BUS (PRIVATE USE)",
        "PRIVATE SERVICE VEHICLE", "PRIVATE SERVICE VEHICLE (INDIVIDUAL USE)",
        "SEMI-TRAILER (COMMERCIAL)", "THREE WHEELER (GOODS)", "THREE WHEELER (PASSENGER)",
        "THREE WHEELER (PERSONAL)", "TRACTOR (COMMERCIAL)"
    ]
    state_ev_groups = ["high_ev_states", "moderate_ev_states", "low_ev_states"]

    start_ref_date = pd.to_datetime("2014-01-01")

    # --- Generate combinations ---
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

    # --- Map category ---
    future_df["Vehicle_Category"] = future_df["Vehicle_Class"].apply(map_vehicle_category)
    future_df["Log_EV_Sales_Quantity"] = np.nan

    # --- Encode and scale ---
    df = future_df.copy()
    for col in ['Vehicle_Class', 'Vehicle_Category']:
        e = LabelEncoder()
        df[col] = e.fit_transform(df[col])

    group_le = joblib.load("src/model/statele.joblib")
    df['State_EV_Group'] = group_le.fit_transform(df['State_EV_Group'])

    scaler = joblib.load("src/model/minmaxscaler.joblib")
    features_to_use = ['Vehicle_Class', 'State_EV_Group', 'Year', 'Vehicle_Category', 'month_sin', 'month_cos']
    X = scaler.fit_transform(df[features_to_use])

    df['Date'] = pd.to_datetime(df['Date'])
    df["Days_Since_Start"] = (df["Date"] - pd.to_datetime("2014-01-01")).dt.days

    sscaler = joblib.load("src/model/standardscaler.joblib")
    df[['Year', 'Month_scaled', 'month_sin', 'month_cos', 'Days_Since_Start']] = sscaler.fit_transform(
        df[['Year', 'Month', 'month_sin', 'month_cos', 'Days_Since_Start']]
    )

    # --- Prepare final data ---
    finalcols = ['Vehicle_Class', 'Vehicle_Category', 'Month_scaled', 'month_sin', 'month_cos',
                 'State_EV_Group', 'Days_Since_Start', 'Log_EV_Sales_Quantity', "Year", "Month"]
    dff = df[finalcols]
    X = dff[['Vehicle_Class', 'Vehicle_Category', 'Month_scaled', 'month_sin', 'month_cos',
             'State_EV_Group', 'Days_Since_Start']].values

    # --- Quantum embedding ---
    dev = qml.device('lightning.qubit', wires=len(X[0]))

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
            embeddings.append(quantum_embedding(row))
        return np.array(embeddings)

    # --- Generate embeddings and predict ---
    X_future_embeddings = generate_embeddings(X)
    model3 = joblib.load("src/model/xgboostqml.joblib")
    y_pred = model3.predict(X_future_embeddings)

    # --- Attach predictions back to data ---
    forecast_df = dff.copy()
    forecast_df["Predicted_Log_EV_Sales_Quantity"] = y_pred

    # --- Return the final DataFrame ---
    result_df = forecast_df[[
        "Month", "State_EV_Group", "Vehicle_Category",
        "Vehicle_Class", "Predicted_Log_EV_Sales_Quantity"
    ]]

    return result_df


