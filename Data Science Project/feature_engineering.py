import numpy as np


class FeatureEngineering:
    def __init__(self, df):
        self.df = df

    def add_features(self):
        print("Feature Engineering Starting...")
        self.df['AvgBedroomsPerRoom'] = self.df['AveBedrms'] / self.df['AveRooms']
        self.df['IncomeByHouseAge'] = self.df['MedInc'] * self.df['HouseAge']
        self.df['Occupancy'] = self.df['AveOccup'] / self.df['AveRooms']
        self.df['PersonsPerBedroom'] = self.df['Occupancy'] / self.df['AveBedrms']

        # Define key points
        key_points = {
            'DistanceToSF': (37.7749, -122.4194),  # San Francisco
            'DistanceToLA': (34.0549, -118.2426),  # Los Angeles
            'DistanceToOcean': (35.6, -120.9),  # Central Coast / Pacific Ocean
            'DistanceToBorder': (32.3, -117.0),  # Southern Border
            'DistanceToNorthernBorder': (42.0, -120.0)  # Northern Border
        }

        # Calculate the distance to key points
        for distance_feature, (lat, lon) in key_points.items():
            self.df[distance_feature] = np.sqrt(
                (self.df['Latitude'] - lat) ** 2 + (self.df['Longitude'] - lon) ** 2)

        print("Feature Engineering Complete")
        return self.df
