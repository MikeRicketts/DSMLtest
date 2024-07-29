import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


class Visualization:
    def __init__(self, df, y_test=None, y_pred=None, rf_model=None, features=None):
        self.df = df
        self.y_test = y_test
        self.y_pred = y_pred
        self.rf_model = rf_model
        self.features = features

    def data_visualization(self):
        print("Data Visualization Starting...")

        # Scaling the house values
        y_test = self.y_test * 100000
        y_pred = self.y_pred * 100000

        # Identify the data ceiling
        ceiling = self.y_test == 5

        # Scatter Plot of Actual vs. Predicted Housing Values
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, edgecolors='k', s=100, c='skyblue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual House Value (in $)')
        plt.ylabel('Predicted House Value (in $)')
        plt.title('Scatter Plot: Actual vs. Predicted House Values')

        # Labeling the Data Ceiling
        plt.scatter(y_test[ceiling], y_pred[ceiling], edgecolors='r', s=100,
                    facecolors='none', linewidths=2, label='Data Ceiling')

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join('Output', 'actual_vs_predicted.png'))
        plt.close()
        print("Scatter Plot saved and closed.")

        # Residual Plot
        residual = y_test - y_pred
        plt.figure(figsize=(12, 8))
        sns.residplot(x=y_test, y=residual, lowess=True, color="g",
                      scatter_kws={'alpha': 0.5, 's': 100, 'edgecolors': 'k', 'color': 'blue'},
                      line_kws={'color': 'red', 'lw': 2})

        # Labeling the Data Ceiling
        plt.scatter(y_test[ceiling], residual[ceiling], edgecolors='r', s=100,
                    facecolors='none', linewidths=2, label='Data Ceiling')

        plt.axhline(0, color='black', linestyle='--', linewidth=2)
        plt.xlabel('Actual House Value (in $)')
        plt.ylabel('Residuals (Difference in $)')
        plt.title('Residual Plot: Difference Between Actual and Predicted Values')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join('Output', 'residual_plot.png'))
        plt.close()
        print("Residual Plot saved and closed.")

        # Feature Importance Plot
        importance = self.rf_model.feature_importances_
        features = np.array(self.features)
        indices = np.argsort(importance)

        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance: Predicting House Values')
        plt.barh(range(len(indices)), importance[indices], color='skyblue', edgecolor='black')
        plt.yticks(range(len(indices)), features[indices])
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.grid(axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join('Output', 'feature_importance.png'))
        plt.close()
        print("Feature Importance Plot saved and closed.")

        print("Data Visualization Complete")
