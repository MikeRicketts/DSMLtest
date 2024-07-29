# Housing Price Prediction Python Final Project

## Overview
The purpose of this project is to develop a predictive model to estimate median house values.
This dataset is originally from SciKit-Learn and contains information about housing in California.
Obtained from Kaggle:
https://www.kaggle.com/datasets/sooyoungher/california-housing

## Libraries Used
- **Pandas:** Data manipulation and analysis.
- **Numpy:** Mathematical operations.
- **Matplotlib:** Data visualization.
- **Seaborn:** Data visualization.
- **Scikit-learn:** Machine learning models and evaluation metrics.
- Required libraries are listed in `requirements.txt`.

## Instructions
1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Run the project with: `python main.py`
3. Expected execution time: ~10 seconds on a fast PC. Tried to optimize the code for faster execution.

## Notes
- Precomputed results and visualizations that I obtained are provided in the 'My Results' folder for reference.

## Project Details

### Main Script (`main.py`)
- **Purpose:** Run the entire project.
- **Steps:**
  - Load the dataset.
  - Clean the data.
  - Perform feature engineering.
  - Visualize the data.
  - Build and evaluate machine learning models.

### Data Cleaning (`data_processing.py`)
- **Purpose:** Ensure the dataset is cleaned up and reliable to work with.
- **Techniques Used:** Dropped rows with missing values and removed duplicate entries.

### Feature Engineering (`feature_engineering.py`)
- **Purpose:** Enhance the dataset with new features for improved model performance.
- **Techniques Used:**
    - Added new features such as persons per bedroom, bedrooms per room, and income by house age.
    - Added geographical features such as distances to major cities, the ocean, and the Southern and Northern borders.

### Data Visualization (`visualization.py`)
- **Purpose:** Understand the data and display the model's performance visually.
- **Techniques Used:**
    - Scatter Plot of Actual vs. Predicted Values
    - Residual Plot of Actual vs. Predicted Values
    - Feature Importance Plot

### Model Building and Evaluation (`model.py`)
- **Purpose:** Develop and evaluate machine learning models.
- **Techniques Used:**
  - Ridge Regression and Random Forest Regressor.
  - Hyperparameter tuning with GridSearchCV.
  - Evaluation metrics: MSE, MAE, and R² score.

## Results
- **Model Performance:**
  - **R² Score:** 0.8217
  - **Mean Squared Error (MSE):** 0.234
  - **Mean Absolute Error (MAE):** 0.305
- **Visualizations:** Saved as image files in the project directory.
- **Metrics:** Saved as text file in the project directory.

## Project Structure
- `main.py`: Main script to run the project.
- `data_processing.py`: Cleans the dataset.
- `feature_engineering.py`: Adds new features to the dataset.
- `visualization.py`: Generates visualizations of the data and model performance.
- `model.py`: Builds and evaluates machine learning model.

## Acknowledgements
- The dataset used in this project is originally from SciKit-Learn and was obtained from Kaggle.
- https://www.kaggle.com/datasets/sooyoungher/california-housing Saved as 'housing.csv' in the project directory.
