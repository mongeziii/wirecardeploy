"""Machine learning module for share price prediction."""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def prepare_features(df):
    """Prepare features for ML models."""
    df = df.copy()
    df = df.sort_values('Date').reset_index(drop=True)

    # Technical indicators
    df['MA_3'] = df['Share_Price'].rolling(3).mean()
    df['MA_6'] = df['Share_Price'].rolling(6).mean()
    df['MA_12'] = df['Share_Price'].rolling(12).mean()
    df['Returns'] = df['Share_Price'].pct_change()
    df['Volatility'] = df['Returns'].rolling(6).std()
    df['Volume_MA'] = df['Volume'].rolling(3).mean()
    df['Price_Change'] = df['Share_Price'].diff()
    df['Momentum'] = df['Share_Price'] - df['Share_Price'].shift(3)

    # Lag features
    for lag in [1, 2, 3, 6]:
        df[f'Lag_{lag}'] = df['Share_Price'].shift(lag)

    # Time features
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Quarter'] = df['Date'].dt.quarter

    # Target: next month price
    df['Target'] = df['Share_Price'].shift(-1)

    return df.dropna()

def train_models(df):
    """Train Random Forest and Gradient Boosting models."""
    df = prepare_features(df)

    feature_cols = [c for c in df.columns if c not in ['Date', 'Target', 'Share_Price']]
    X = df[feature_cols]
    y = df['Target']

    # Split: train on first 80%, test on last 20%
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_test = df['Date'].iloc[split_idx:].values
    actual_test = df['Share_Price'].iloc[split_idx:].values

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)

    # Calculate metrics
    def calc_metrics(actual, predicted):
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100

        # Directional accuracy
        actual_dir = np.sign(np.diff(actual))
        pred_dir = np.sign(np.diff(predicted))
        dir_acc = np.mean(actual_dir == pred_dir) * 100 if len(actual_dir) > 0 else 0

        return {'MAE': round(mae, 2), 'RMSE': round(rmse, 2), 'R2': round(r2, 3), 
                'MAPE': round(mape, 2), 'Directional_Accuracy': round(dir_acc, 1)}

    rf_metrics = calc_metrics(y_test, rf_pred)
    gb_metrics = calc_metrics(y_test, gb_pred)

    # Feature importance
    rf_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)

    return {
        'RandomForest': {
            'metrics': rf_metrics,
            'predictions': rf_pred,
            'model': rf,
            'importance': rf_importance
        },
        'GradientBoosting': {
            'metrics': gb_metrics,
            'predictions': gb_pred,
            'model': gb
        },
        'test_data': {
            'dates': dates_test,
            'actual': y_test.values,
            'actual_current': actual_test,
            'X_test': X_test
        },
        'feature_cols': feature_cols
    }

def get_best_model(results):
    """Determine best performing model."""
    rf_r2 = results['RandomForest']['metrics']['R2']
    gb_r2 = results['GradientBoosting']['metrics']['R2']

    if rf_r2 > gb_r2:
        return 'Random Forest', results['RandomForest']['metrics']
    else:
        return 'Gradient Boosting', results['GradientBoosting']['metrics']


def train_linear_regression_baseline(df):
    """Train a simple Linear Regression baseline model."""
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    df = prepare_features(df)
    feature_cols = [c for c in df.columns if c not in ['Date', 'Target', 'Share_Price']]
    X = df[feature_cols]
    y = df['Target']

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_test = df['Date'].iloc[split_idx:].values
    actual_test = df['Share_Price'].iloc[split_idx:].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    mae = mean_absolute_error(y_test, lr_pred)
    rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
    r2 = r2_score(y_test, lr_pred)
    mape = np.mean(np.abs((y_test - lr_pred) / y_test)) * 100

    actual_dir = np.sign(np.diff(y_test.values))
    pred_dir = np.sign(np.diff(lr_pred))
    dir_acc = np.mean(actual_dir == pred_dir) * 100 if len(actual_dir) > 0 else 0

    return {
        'metrics': {'MAE': round(mae, 2), 'RMSE': round(rmse, 2), 'R2': round(r2, 3), 
                    'MAPE': round(mape, 2), 'Directional_Accuracy': round(dir_acc, 1)},
        'predictions': lr_pred,
        'model': lr,
        'test_data': {'dates': dates_test, 'actual': y_test.values, 'actual_current': actual_test}
    }
