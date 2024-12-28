MLB OPS PREDICTION MODEL
Overview:
        This project predicts whether a baseball player's OPS (On-base Plus Slugging) will exceed 0.800 in 2025 based on their 
        statistics from the 2023, and 2024 MLB seasons. Using AdaBoost with a DecisionTreeClassifier as the base estimator, 
        this model leverages historical player data for feature extraction and prediction.

Features:
        - Machine Learning Model: Implements AdaBoost with hyperparameter tuning for learning rates.
        - Interactive Prediction: Allows users to input a player ID via the terminal to predict their OPS for 2025.
        - Visualization: Plots training and test accuracies against different learning rates.
        - Data Handling: Processes MLB data stored in an SQLite database and structures it for model training.

Technologies Used:
        - Programming Language: Python
Libraries: 
        - pandas for data manipulation
        - numpy for numerical operations
        - sklearn for machine learning
        - matplotlib for data visualization
        - sqlite3 for database integration

HOW IT WORKS
Data Preparation:
        1. Retrieves player statistics from an SQLite database built via free MLBSTATS API.
        2. Pivots data to align stats from 2022 and 2023 for each player.
        3. Merges 2024 statistics and creates a binary target variable (OPS_above_800).
Model Training:
        1. Trains an AdaBoost classifier with a DecisionTreeClassifier as the base estimator.
        2. Tests model performance on unseen data.
        3. Optimizes the learning rate for the AdaBoost algorithm.
Interactive Prediction:
        - Prompts the user to input a player ID and uses the trained model to predict the player's OPS performance in 2025.
