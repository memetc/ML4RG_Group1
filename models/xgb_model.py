import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class XGBoostModel:
    '''
        XGBoost model for predicting TPM values.
        Input: (species, stress_condition, base_sequence)
        Each one-hot encoded.
        Output: predicted TPM value for the given input.
    '''
    
    def __init__(self, **kwargs):
        # hyperparameters
        self.species_size = kwargs['species_size'] if 'species_size' in kwargs else 30
        self.stress_condition_size = kwargs['stress_condition_size'] if 'stress_condition_size' in kwargs else 12
        self.hidden_size = kwargs['hidden_size'] if 'hidden_size' in kwargs else 30
        
        # XGBoost parameters
        self.params = {
            'objective': 'reg:squarederror',
            'max_depth': kwargs['max_depth'] if 'max_depth' in kwargs else 6,
            'learning_rate': kwargs['lr'] if 'lr' in kwargs else 0.1,
            'n_estimators': kwargs['n_estimators'] if 'n_estimators' in kwargs else 100,
            'verbosity': 1
        }
        self.model = xgb.XGBRegressor(**self.params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        mse = mean_squared_error(y, predictions)
        return mse