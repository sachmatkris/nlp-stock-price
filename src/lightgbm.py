import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

INPUT_PATH = 'df_disney_final.csv'
LGBM_PARAMS = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.01,
        'num_leaves': 31,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbosity': -1,
        'seed': 42
    }

if __name__ == '__main__':
    df_stock = pd.read_csv(INPUT_PATH)
    df_stock['tomorrow_pct_return'] = df_stock.loc[:,'pct_return'].shift(1).drop(columns='news').dropna(axis=0)
    df_stock = df_stock.loc[(df_stock['sentiment_score'] > 0.5) | (df_stock['sentiment_score'] < -0.5), :] # You can change the sen timent thresholds

    X_lightgbm, Y_lightgbm = df_stock.drop(columns='tomorrow_pct_return'), df_stock['tomorrow_pct_return']
    X_train, X_val, y_train, y_val = train_test_split(X_lightgbm, Y_lightgbm, test_size=0.2, shuffle=False)

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

    model = lgb.train(
        LGBM_PARAMS,
        train_data,
        valid_sets=[train_data, val_data],
        num_boost_round=1000,
    )
    model.save_model("lightgbm_model.txt")

    # Evaluation
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    directional_acc = (np.sign(y_pred) == np.sign(y_val)).mean()

    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R²: {r2:.6f}")
    print(f"Directional Accuracy: {directional_acc * 100:.2f}%")