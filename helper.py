import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import lightgbm as lgb
import optuna
import pickle

def label_encode_categorical(df, categorical_cols) -> pd.DataFrame:
    le = LabelEncoder()
    
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))  # Ensure columns are treated as strings
    return df

def create_lags(df: pd.DataFrame, lag_days: int = 28) -> pd.DataFrame:
    df = df.sort_values(by=["id", "date"])

    for lag in range(1, lag_days + 1):
        df[f'lag_{lag}'] = df.groupby(['id'])['y'].shift(lag)

    return df

def create_date_features(df, date_col):
    df[date_col] = pd.to_datetime(df[date_col])
    df['day'] = df[date_col].dt.day
    df['month'] = df[date_col].dt.month
    df['year'] = df[date_col].dt.year
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['is_weekend'] = df[date_col].dt.dayofweek >= 5
    return df

def convert_to_category(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype('category')
        else:
            print(f"Warning: Column '{col}' not found in DataFrame.")
    
    return df

# Optuna optimization
def optuna_process(X_train, X_test, y_train, y_test):
    def objective(trial):
        
        params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300)
            }
        
        model = lgb.LGBMRegressor(**params, n_jobs=-1, random_state=42)
        
        model.set_params(**{
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'feature_pre_filter': False,
            'force_col_wise': True,
            'force_row_wise': False,
            'nan_mode': 'min',
            'categorical_feature': 'auto'
        })
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return rmse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)

    print('Best trial:')
    trial = study.best_trial
    print(f'  Value: {trial.value}')
    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

    best_model = lgb.LGBMRegressor(**trial.params)

    best_model.fit(X_train, y_train)

    # Serialize the best model
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    return best_model

features = [f'sales_lag_{lag}' for lag in range(1, 29)] + ['sell_price', 'day', 'month', 'year', 'dayofweek', 'is_weekend'] 
target = 'y'

# Create 28-day forecast function
def create_forecast(df, model):
    forecast_dates = [f'd_{i}' for i in range(1942, 1969)]  # Forecast for d_1942 to d_1969
    future_data = df[df['d'].isin(forecast_dates)]

    X_future = future_data[features]
    y_pred_future = model.predict(X_future)

    # Return the forecasted sales
    future_data['forecast_sales'] = y_pred_future
    return future_data[['id', 'd', 'forecast_sales']]