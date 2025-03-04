import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import lightgbm as lgb
import optuna
import pickle
import gc
import dask.dataframe as dd

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
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, 0.5),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100, 300, 500 )
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
    study.optimize(objective, n_trials=50)

    print('Best trial:')
    trial = study.best_trial
    print(f'  Value: {trial.value}')
    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

    best_model = lgb.LGBMRegressor(**trial.params)

    best_model.fit(X_train, y_train)

    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    return best_model

def create_forecast(df, model_name, features):
    model = pickle.load(open(model_name, 'rb'))

    y_pred_future = model.predict(df)

    df['forecast_sales'] = y_pred_future

    future_data = df.drop(columns=['wm_yr_wk', 'event_name_1',  'event_type_1',  'event_name_2',  'event_type_2',  'snap_CA',
                                             'snap_TX',  'snap_WI',  'sell_price','day',  'month',  'year',  'dayofweek',  'is_weekend'])  # Drop the feature columns
    
    gc.collect()

    future_data['forecast_sales'] = round(future_data['forecast_sales'].astype(int))
    future_data = future_data.reset_index()
    original_data = future_data[['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].copy()


    df_pivot = future_data.pivot(index=['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], 
                    columns='d', 
                    values='forecast_sales').reset_index()
    
    df_pivot = original_data.merge(df_pivot, on=['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], how='left')
    
    return df_pivot
