import pandas as pd
import os
from data_processing import DataProcessing
from feature_engineering import FeatureEngineering
from visualization import Visualization
from model import Model

# User Input to save model metrics
metrics_file = (input("Enter the name for the model's metrics file (i.e 'metrics.txt') or press Enter to use default: ")
                .strip())
if not metrics_file:
    metrics_file = "default_metrics.txt"
elif not metrics_file.endswith('.txt'):
    metrics_file += '.txt'

# Save the metrics file in the Output folder
metrics_file = os.path.join('Output', metrics_file)

# Read the data
df = pd.read_csv('housing.csv')  # Dataset from https://www.kaggle.com/datasets/sooyoungher/california-housing

# Preprocessing the data
data_process = DataProcessing(df)
df_cleaned = data_process.data_cleaning()

# Feature engineering
feature_engineer = FeatureEngineering(df_cleaned)
df_features = feature_engineer.add_features()

# Model building
model_builder = Model(df_features)
y_test, y_pred, rf_model, features = model_builder.build_model(metrics_file)

# Data visualization
visualizer = Visualization(df_features, y_test, y_pred, rf_model, features)
visualizer.data_visualization()

print(f"Model metrics have been saved as '{metrics_file}'.")
