import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class Model:
    def __init__(self, df):
        self.df = df

    def build_model(self, filename):
        print("Building Model...")

        # One-hot encoding
        self.df = pd.get_dummies(self.df)

        # Features to use in the model
        features = ['MedInc', 'AvgBedroomsPerRoom', 'IncomeByHouseAge',
                    'DistanceToSF', 'DistanceToOcean', 'DistanceToBorder', 'DistanceToLA',
                    'DistanceToNorthernBorder', 'Occupancy', 'PersonsPerBedroom']

        # Extract features and target variable
        x = self.df[features]
        y = self.df['MedHouseVal']

        # Check for missing values
        if x.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
            raise Exception("Missing values in the dataset.")

        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Standardize the features
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        # Ridge Regression with cross-validation
        ridge = Ridge()
        grid_parameters = {'alpha': [0.1, 1, 10, 100, 1000]}
        grid_search = GridSearchCV(ridge, grid_parameters, cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(x_train_scaled, y_train)

        # Random Forest Regressor model
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(x_train, y_train)

        # Predictions on the test set
        y_pred = rf.predict(x_test)

        # Performance metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("Model Building Complete")

        # Save Metrics to a text file
        with open(filename, "w") as file:
            file.write(f"Mean Squared Error: {mse}\n")
            file.write(f"Mean Absolute Error: {mae}\n")
            file.write(f"R2 Score: {r2}\n")
            file.write(f"Best Ridge Parameters: {grid_search.best_params_}\n")

        return y_test, y_pred, rf, features
