"""
XGBoost Machine Learning Time-Series Forecaster Module
Gradient-boosted decision tree regression with feature importance and lag dynamics.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import xgboost as xgb
from ..feature_engineering import build_full_feature_matrix


class XGBoostForecaster:
    """
    XGBoost Time-Series Regressor with recursive lag feature updates.
    """
    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

        self.model: Optional[xgb.XGBRegressor] = None
        self.feature_names: List[str] = []
        self.train_dates: Optional[pd.DatetimeIndex] = None
        self.full_train_df: Optional[pd.DataFrame] = None
        self.residual_std: float = 0.0

    def fit(self, df_features: pd.DataFrame, target_col: str = 'sales') -> 'XGBoostForecaster':
        """
        Fit XGBoost regressor on feature matrix.
        """
        self.full_train_df = df_features.copy()
        self.train_dates = df_features.index

        feature_cols = [c for c in df_features.columns if c not in [target_col, 'gross_sales', 'transactions', 'avg_order_value', 'discount_amount']]
        self.feature_names = feature_cols

        X = df_features[feature_cols].values
        y = df_features[target_col].values

        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X, y)

        # In-sample residuals
        in_sample_preds = self.model.predict(X)
        residuals = y - in_sample_preds
        self.residual_std = float(np.std(residuals))

        return self

    def predict(
        self,
        steps: int,
        df_future_calendar: Optional[pd.DataFrame] = None,
        freq: str = 'D'
    ) -> pd.DataFrame:
        """
        Predict future steps using recursive feature updates.
        """
        if self.model is None or self.full_train_df is None:
            raise ValueError("Model has not been fitted.")

        history_df = self.full_train_df.copy()
        last_date = self.train_dates[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq=freq)

        yhat_list = []
        
        for cur_date in future_dates:
            # Rebuild features for current horizon step
            feat_df = build_full_feature_matrix(history_df, drop_na_rows=False)
            current_row_features = feat_df[self.feature_names].iloc[-1:].values
            
            # Predict
            pred_val = float(self.model.predict(current_row_features)[0])
            pred_val = max(0.0, pred_val)
            yhat_list.append(pred_val)

            # Append new predicted row to history for next recursive lag calculation
            new_row = pd.DataFrame({
                'sales': [pred_val],
                'discount_rate': [history_df['discount_rate'].mean() if 'discount_rate' in history_df else 18.0]
            }, index=[cur_date])
            history_df = pd.concat([history_df, new_row])

        yhat = np.array(yhat_list)
        uncertainty = np.sqrt(np.arange(1, steps + 1))
        lower_bound = np.clip(yhat - 1.96 * self.residual_std * (0.8 + 0.1 * uncertainty), 0, None)
        upper_bound = np.clip(yhat + 1.96 * self.residual_std * (0.8 + 0.1 * uncertainty), 0, None)

        forecast_df = pd.DataFrame({
            'yhat': yhat,
            'yhat_lower': lower_bound,
            'yhat_upper': upper_bound
        }, index=future_dates)
        forecast_df.index.name = 'Date'
        return forecast_df

    def get_feature_importances(self) -> pd.DataFrame:
        """
        Return sorted feature importances.
        """
        if self.model is None:
            return pd.DataFrame()
        importances = self.model.feature_importances_
        df_imp = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).reset_index(drop=True)
        return df_imp
