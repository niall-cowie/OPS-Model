import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import sqlite3
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class AdaBoostMLBModel:
    def __init__(self, db_path, learning_rates=None, test_size=0.2, random_state=42):
        # Initialization parameters
        self.db_path = db_path
        self.learning_rates = learning_rates if learning_rates else [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
        self.test_size = test_size
        self.random_state = random_state

        # Placeholder for the data
        self.df = None
        self.df_2022_2023_pivot = None
        self.df_2024_merged = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        # Load data from the database
        conn = sqlite3.connect(self.db_path)
        #confirming we are only grabbing players who've played in 2022, 2023, and 2024
        query = """
        SELECT *
        FROM player_stats
        WHERE player_id IN (
            SELECT player_id
            FROM player_stats
            GROUP BY player_id
            HAVING COUNT(DISTINCT year) = 3
        )
        """
        self.df = pd.read_sql(query, conn)
        conn.close()

    def prepare_data(self):
        # Prepare the data for training and testing
        df_2022_2023 = self.df[self.df['year'].isin([2022, 2023])]
        
        # Pivot the data to have one row per player with stats from both 2022 and 2023
        self.df_2022_2023_pivot = df_2022_2023.pivot_table(
            index='player_id',
            columns='year',
            values=['ops', 'atbats', 'hits', 'homeruns', 'avg', 'baseonballs'],
            aggfunc='first'
        )
        
        # Flatten the columns after pivoting
        self.df_2022_2023_pivot.columns = [f"{col[0]}_{col[1]}" for col in self.df_2022_2023_pivot.columns]
        
        # Extract data for 2024
        df_2024 = self.df[self.df['year'] == 2024]
        
        # Merge 2024 data with 2022-2023 stats
        self.df_2024_merged = pd.merge(df_2024, self.df_2022_2023_pivot, on='player_id', how='left')
        
        # Create the target column for predicting OPS above 0.800 in 2024
        self.df_2024_merged['OPS_above_800'] = (self.df_2024_merged['ops'] >= 0.800).astype(int)

    def split_data(self):
        # Split the data into training and testing sets
        X = self.df_2022_2023_pivot.drop(columns=['ops_2022', 'ops_2023'])
        y = self.df_2024_merged['OPS_above_800']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state
        )

    def train_model(self):
        # Create and train the AdaBoost classifier for different learning rates
        train_accuracies = []
        test_accuracies = []

        for learning_rate in self.learning_rates:
            adaboost = AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1),
                n_estimators=50,
                learning_rate=learning_rate,
                random_state=self.random_state
            )
            adaboost.fit(self.X_train, self.y_train)
            
            # Predict and calculate accuracies
            y_train_pred = adaboost.predict(self.X_train)
            train_accuracies.append(accuracy_score(self.y_train, y_train_pred))
            
            y_test_pred = adaboost.predict(self.X_test)
            test_accuracies.append(accuracy_score(self.y_test, y_test_pred))

        return train_accuracies, test_accuracies

    def plot_results(self, train_accuracies, test_accuracies):
        # Plot the learning rates vs accuracies
        plt.plot(self.learning_rates, train_accuracies, label='Train Accuracy', marker='o')
        plt.plot(self.learning_rates, test_accuracies, label='Test Accuracy', marker='o')
        plt.xlabel('Learning Rate')
        plt.ylabel('Accuracy')
        plt.title('Train and Test Accuracy vs Learning Rate for AdaBoost')
        plt.legend()
        plt.grid(True)
        plt.show()

    def predict_2025_ops(self, player_id):
        """
        Given a player's ID, predict if their OPS in 2025 will be above or below 0.800.
        The prediction is based on their stats from 2023 and 2024.
        """
        # Get the player's 2023 and 2024 statistics
        player_data = self.df_2022_2023_pivot.loc[[player_id]].copy()
        
        # Check if player data exists
        if player_data.empty:
            print(f"Player ID {player_id} not found in the dataset.")
            return
        
        # Use the trained AdaBoost model to predict the OPS above 0.800 for 2025
        adaboost = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=50,
            learning_rate=.5,  # lr=.5 (relatively low E_approx)
            random_state=self.random_state
        )
        adaboost.fit(self.X_train, self.y_train)

        # Predict the player's OPS above 0.800
        player_prediction = adaboost.predict(player_data.drop(columns=['ops_2022', 'ops_2023']))
        
        if player_prediction == 1:
            print(f"Player ID {player_id} is predicted to have an OPS above 0.800 in 2025.")
        else:
            print(f"Player ID {player_id} is predicted to have an OPS below 0.800 in 2025.")

    def run(self):
        # Run the whole process
        self.load_data()
        self.prepare_data()
        self.split_data()
        train_accuracies, test_accuracies = self.train_model()
        self.plot_results(train_accuracies, test_accuracies)

if __name__ == "__main__":
    db = 'mlb_2022-2024_stats.db'
    model = AdaBoostMLBModel(db)
    model.run()
    
    # Ask user for a player ID and predict their 2025 OPS
    player_id = int(input("Enter Player ID to predict their OPS for 2025: "))
    model.predict_2025_ops(player_id)
