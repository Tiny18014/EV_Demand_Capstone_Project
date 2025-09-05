import joblib
import pandas as pd
import numpy as np
import pennylane as qml
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import sys


#maps
high_ev_states = ['Delhi', 'Maharashtra', 'Karnataka', 'Tamil Nadu', 'Telangana', 'Gujarat']
moderate_ev_states = ['Kerala', 'Andhra Pradesh', 'Haryana', 'Rajasthan', 'Punjab', 'West Bengal']
low_ev_states = ['Uttar Pradesh', 'Bihar', 'Jharkhand', 'Madhya Pradesh', 'Odisha', 'Assam', 'Chhattisgarh',
                 'Uttarakhand', 'Himachal Pradesh', 'Goa', 'Jammu and Kashmir', 'Tripura', 'Nagaland', 'Meghalaya', 'Others']

#functions
def map_vehicle_category(vehicle: str) -> str:
  v = vehicle.upper()
  # 2-WHEELERS
  if any(x in v for x in ["M-CYCLE", "MOTOR CYCLE", "MOTORISED CYCLE", "SCOOTER", "MOPED"]):
    return "2-wheelers"
  # 3-WHEELERS
  if "THREE WHEELER" in v or "E-RICKSHAW" in v or "QUADRICYCLE" in v:
    return "3-wheelers"
  # 4-WHEELERS (cars, cabs, vans, personal service vehicles)
  if any(x in v for x in ["MOTOR CAR", "MOTOR CAB", "MAXI CAB", "LUXURY CAB", "CASH VAN",
  "PRIVATE SERVICE VEHICLE", "OMNI BUS (PRIVATE USE)",
  "MOTOR CARAVAN", "CAMPER VAN", "HEARSES"]):
    return "4-wheelers"
  # BUS
  if "BUS" in v:
    return "bus"
  # EVERYTHING ELSE
  return "other"
def group_state(state):
    if state in high_ev_states:
        return 'High_EV'
    elif state in moderate_ev_states:
        return 'Moderate_EV'
    else:
        return 'Low_EV'


if __name__ == "__main__":
    dataset_path = sys.argv[1]
    df = pd.read_csv(dataset_path, encoding="cp1252", skiprows=2)
    print(f"Loaded dataset: {dataset_path} with shape {df.shape}")
    #excel to csv not done yet, add it here, and indent everything else
    df = df.rename(columns={
        df.columns[0]: "S_No",
        df.columns[1]: "Vehicle_Class"
    })

    # Melt from wide to long format
    df_long = df.melt(
        id_vars=["S_No", "Vehicle_Class"],
        var_name="Month",
        value_name="EV_Sales_Quantity"
    )

    # Clean thousands separators and convert to numeric
    df_long["EV_Sales_Quantity"] = (
        df_long["EV_Sales_Quantity"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(int)
    )

    # Add context columns
    df_long["Year"] = 2025
    df_long["State"] = "Karnataka"

    # Reorder final columns
    df_final = df_long[["Month", "Year", "State", "EV_Sales_Quantity", "Vehicle_Class"]]
    df_final["Vehicle_Category"] = df_final["Vehicle_Class"].apply(map_vehicle_category)
    df = df_final


    #code
    df['Month'] = df['Month_Name'].map({'jan':1, 'feb':2, 'mar':3, 'apr':4, "may":5, "jun":6, "jul":7,"sep":9, "aug":8, "oct":10, "nov":11 ,'dec':12})
    df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    categorical_cols = ['Vehicle_Class', 'Vehicle_Category']
    for col in categorical_cols:
        e = LabelEncoder()
        df[col] = e.fit_transform(df[col])
    df['State_EV_Group'] = df['State'].apply(group_state)
    group_le = LabelEncoder()
    df['State_EV_Group'] = group_le.fit_transform(df['State_EV_Group'])
    scaler = MinMaxScaler(feature_range=(-1, 1))
    features_to_use = ['Vehicle_Class', 'State_EV_Group', 'Year',
                      'Vehicle_Category','month_sin', 'month_cos']

    X = scaler.fit_transform(df[features_to_use])
    df['Date'] = pd.to_datetime(df['Date'])
    df['Log_EV_Sales_Quantity'] = np.log1p(df['EV_Sales_Quantity'])
    df['Days_Since_Start'] = (df['Date'] - df['Date'].min()).dt.days
    scaler = StandardScaler()
    df[['Year', 'Month', 'month_sin', 'month_cos', 'Days_Since_Start']] = scaler.fit_transform(df[['Year', 'Month', 'month_sin', 'month_cos', 'Days_Since_Start']])
    finalcols = ['Vehicle_Class', 'Vehicle_Category','Month', 'month_sin', 'month_cos', 'State_EV_Group', 'Days_Since_Start','Log_EV_Sales_Quantity' ]
    dff = df[finalcols]
    og = pd.read_csv("src/data/qmldata/ready_dup.csv")
    og = pd.concat([og, dff], ignore_index=True)
    og.to_csv("src/data/qmldata/ready_dup.csv")


