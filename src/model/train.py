from datetime import datetime, timedelta
import pickle
from river import metrics
from src.data.preprocess import preprocess  

today = datetime.today()

# Set end date as **yesterday** (i.e., last day of the previous month)
end_date = today
start_date = end_date - timedelta(days=7)

start_str = start_date.strftime('%Y-%m-%d')
end_str = end_date.strftime('%Y-%m-%d')

# Step 3: Preprocess data using dynamic date strings
x, y = preprocess(start_str, end_str)
x.to_csv("data/feat.csv", index=False)
y.to_csv("data/targ.csv", index=False)
# Step 4: Load existing model
with open("src/model/rm1.pkl", "rb") as f:
    model = pickle.load(f)

# Step 5: Train model incrementally
for xi, yi in zip(x.to_dict(orient="records"), y):
    if yi is not None:
        model.learn_one(xi, yi)

# Step 6: Evaluate updated model
metric = metrics.Accuracy()
for xi, yi in zip(x.to_dict(orient="records"), y):
    if yi is not None:
        y_pred = model.predict_one(xi)
        if y_pred is not None:
            metric.update(yi, y_pred)

# Step 7: Save updated model
with open("src/model/rm1.pkl", "wb") as f:
    pickle.dump(model, f)

# Step 8: Print evaluation result
print(f"Training from {start_str} to {end_str}: Accuracy = {metric.get():.4f}")