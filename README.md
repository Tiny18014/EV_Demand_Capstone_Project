# Forecasting Electric Vehicle Demand: Sentiment, Charging Behavior & Quantum vs Classical Machine Learning

## Sentiment Analysis
Data extracted and preprocessed weekly, automated by cron jobs and a CI/CD pipeline is fed to an online learning River model (Leveraging Bagging Classifier, with a Hoeffding Tree Classifier) which trains incrementally. The newest model is then fetched in the Streamlit application, to perform a competitor brand sentiment forecast for the next day.

## Hybrid Quantum-Classical Framework for Demand Forecasting
Data downloaded from the Vahan Portal is pushed to this pipeline, which preprocesses it, and simulates updated training of an XGBoost model. This model then forecasts quarterly EV sales across high, medium and low concentration EV state categories, and vehicle categories.