import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import pickle
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('cbr_clean_data.csv')
X = df.drop('CBR_soaked', axis=1)
y = df['CBR_soaked']

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = np.abs(y_true) > 0.1
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

mape_scorer = make_scorer(mape, greater_is_better=False)

rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)

scoring = {
    'r2': 'r2',
    'neg_rmse': 'neg_root_mean_squared_error',
    'neg_mae': 'neg_mean_absolute_error',
    'neg_mape': mape_scorer
}

# models that need scaling get a pipeline
SCALE = {'ElasticNet', 'SVR', 'GPR', 'ANN'}

models = {
    'ElasticNet':   ElasticNet(random_state=42, max_iter=10000),
    'RandomForest': RandomForestRegressor(random_state=42, n_jobs=-1),
    'ExtraTrees':   ExtraTreesRegressor(random_state=42, n_jobs=-1),
    'XGBoost':      XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
    'LightGBM':     LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1),
    'CatBoost':     CatBoostRegressor(random_state=42, verbose=False, allow_writing_files=False),
    'SVR':          SVR(),
    'GPR':          GaussianProcessRegressor(
                        kernel=ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1.0),
                        random_state=42, n_restarts_optimizer=5),
    'ANN':          MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000,
                                  random_state=42, early_stopping=True),
}

results = {}
detailed = {}

for name, model in models.items():
    est = Pipeline([('scaler', StandardScaler()), ('model', model)]) if name in SCALE else model
    try:
        cv = cross_validate(est, X, y, cv=rkf, scoring=scoring, n_jobs=-1)
        results[name] = {
            'R2_mean':   cv['test_r2'].mean(),      'R2_std':   cv['test_r2'].std(),
            'RMSE_mean': -cv['test_neg_rmse'].mean(),'RMSE_std': cv['test_neg_rmse'].std(),
            'MAE_mean':  -cv['test_neg_mae'].mean(), 'MAE_std':  cv['test_neg_mae'].std(),
            'MAPE_mean': -cv['test_neg_mape'].mean(),'MAPE_std': cv['test_neg_mape'].std(),
        }
        detailed[name] = {
            'R2':   cv['test_r2'],
            'RMSE': -cv['test_neg_rmse'],
            'MAE':  -cv['test_neg_mae'],
            'MAPE': -cv['test_neg_mape'],
        }
        print(f"{name}: R²={results[name]['R2_mean']:.4f} ± {results[name]['R2_std']:.4f}")
    except Exception as e:
        print(f"{name} failed: {e}")

results_df = pd.DataFrame(results).T.round(4).sort_values('R2_mean', ascending=False)
print(results_df)
results_df.to_csv('model_cv_results.csv')

with open('detailed_cv_results.pkl', 'wb') as f:
    pickle.dump(detailed, f)
