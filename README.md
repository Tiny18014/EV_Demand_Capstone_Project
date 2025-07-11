# Forecasting Electric Vehicle Demand: Sentiment, Charging Behavior & Quantum vs Classical Machine Learning

## Sentiment Analysis
Data extracted and preprocessed weekly, automated by cron jobs and a CI/CD pipeline is fed to an online learning River model (Leveraging Bagging Classifier, with a Hoeffding Tree Classifier) which trains incrementally. The newest model is then fetched in the Streamlit application, to perform a competitor brand sentiment forecast for the next day.

