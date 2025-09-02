import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
import pennylane as qml
import numpy as np
import pandas as pd
import joblib


df = pd.read_csv("src/data/qmldata/ready.csv")
X = df[['Vehicle_Class', 'Vehicle_Category', 'Month', 'month_sin', 'month_cos', 'State_EV_Group', 'Days_Since_Start']].values
y = df['Log_EV_Sales_Quantity'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dev = qml.device('lightning.qubit', wires=len(X_train[0]))  # One wire per feature

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

X_train_embeddings = generate_embeddings(X_train)
X_test_embeddings = generate_embeddings(X_test)
print("embeddings done")

model3 = xgb.XGBRegressor(objective='reg:squarederror',
                         n_estimators=621,
                         learning_rate=0.11031768290348885,
                         max_depth=6,
                         subsample=1.0,
                         colsample_bytree=1.0,
                         random_state=42)

model3.fit(X_train_embeddings, y_train)

# Make predictions using the trained model
y_pred = model3.predict(X_test_embeddings)

# Evaluate the performance using Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (XGBoost with Quantum Embeddings): {mse}")

# Optional: Use RMSE for better interpretability
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse}")

rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
print(f"RMSLE: {rmsle}")

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
r2 = r2_score(y_test, y_pred)
print(f"R2 Score: {r2}")

joblib.dump(model3, "src/model/qml_model.joblib")