import pandas as pd
import numpy as np
import pickle, json, time, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import RandomizedSearchCV, KFold, RepeatedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from scipy.stats import randint, uniform

SEED     = 42
N_ITER   = 50
INNER_CV = KFold(n_splits=5, shuffle=True, random_state=SEED)

df = pd.read_csv('cbr_clean_data.csv')
X  = df.drop('CBR_soaked', axis=1)
y  = df['CBR_soaked']

# quintile binning for outer CV stratification
y_binned = pd.qcut(y, q=5, labels=False, duplicates='drop')

search_spaces = {
    'ElasticNet': {
        'model': ElasticNet(random_state=SEED, max_iter=5000),
        'params': {
            'model__alpha':    uniform(1e-4, 10),
            'model__l1_ratio': uniform(0.1, 0.9),
        }
    },
    'SVR': {
        'model': SVR(),
        'params': {
            'model__C':       uniform(0.1, 999.9),
            'model__gamma':   ['scale', 'auto'] + list(uniform(0.001, 0.999).rvs(20, random_state=SEED)),
            'model__epsilon': uniform(0.01, 0.49),
            'model__kernel':  ['rbf'],
        }
    },
    'GPR': {
        'model': GaussianProcessRegressor(random_state=SEED, normalize_y=True),
        'params': {
            'model__kernel':               [RBF(), Matern(nu=1.5), Matern(nu=2.5)],
            'model__alpha':                uniform(1e-10, 0.1),
            'model__n_restarts_optimizer': randint(0, 10),
        }
    },
    'RandomForest': {
        'model': RandomForestRegressor(random_state=SEED, n_jobs=-1),
        'params': {
            'model__n_estimators':      randint(50, 501),
            'model__max_depth':         [None] + list(randint(5, 31).rvs(15, random_state=SEED)),
            'model__min_samples_split': randint(2, 21),
            'model__min_samples_leaf':  randint(1, 9),
            'model__max_features':      ['sqrt', 'log2', None, 0.5, 0.7],
        }
    },
    'ExtraTrees': {
        'model': ExtraTreesRegressor(random_state=SEED, n_jobs=-1),
        'params': {
            'model__n_estimators':      randint(50, 501),
            'model__max_depth':         [None] + list(randint(5, 31).rvs(15, random_state=SEED)),
            'model__min_samples_split': randint(2, 21),
            'model__min_samples_leaf':  randint(1, 9),
            'model__max_features':      ['sqrt', 'log2', None, 0.5, 0.7],
        }
    },
    'XGBoost': {
        'model': XGBRegressor(random_state=SEED, n_jobs=-1, verbosity=0,
                              objective='reg:squarederror'),
        'params': {
            'model__n_estimators':      randint(100, 1001),
            'model__max_depth':         randint(3, 11),
            'model__learning_rate':     uniform(0.01, 0.29),
            'model__subsample':         uniform(0.5, 0.5),
            'model__colsample_bytree':  uniform(0.5, 0.5),
            'model__colsample_bylevel': uniform(0.5, 0.5),
            'model__colsample_bynode':  uniform(0.5, 0.5),
            'model__reg_alpha':         uniform(0.0, 1.0),
            'model__reg_lambda':        uniform(0.5, 2.0),
            'model__min_child_weight':  randint(1, 11),
            'model__gamma':             uniform(0.0, 0.5),
            'model__max_delta_step':    randint(0, 6),
        }
    },
    'LightGBM': {
        'model': LGBMRegressor(random_state=SEED, n_jobs=-1, verbosity=-1),
        'params': {
            'model__n_estimators':      randint(100, 1001),
            'model__max_depth':         randint(3, 11),
            'model__learning_rate':     uniform(0.01, 0.29),
            'model__num_leaves':        randint(20, 151),
            'model__subsample':         uniform(0.5, 0.5),
            'model__colsample_bytree':  uniform(0.5, 0.5),
            'model__reg_alpha':         uniform(0.0, 1.0),
            'model__reg_lambda':        uniform(0.0, 1.0),
            'model__min_child_samples': randint(5, 51),
        }
    },
    'CatBoost': {
        'model': CatBoostRegressor(random_seed=SEED, verbose=0, thread_count=-1),
        'params': {
            'model__iterations':        randint(100, 1001),
            'model__depth':             randint(4, 11),
            'model__learning_rate':     uniform(0.01, 0.29),
            'model__l2_leaf_reg':       uniform(1.0, 9.0),
            'model__subsample':         uniform(0.5, 0.5),
            'model__colsample_bylevel': uniform(0.5, 0.5),
        }
    },
    'ANN': {
        'model': MLPRegressor(random_state=SEED, max_iter=500,
                              early_stopping=True, n_iter_no_change=20),
        'params': {
            'model__hidden_layer_sizes': [(64,32), (128,64), (64,32,16), (32,16), (100,50)],
            'model__learning_rate_init': uniform(1e-4, 1e-2),
            'model__alpha':              uniform(1e-5, 1e-2),
            'model__batch_size':         [16, 32, 64],
            'model__activation':         ['relu', 'tanh'],
        }
    },
}

tuning_results = {}
best_models    = {}

for name, cfg in search_spaces.items():
    t0   = time.time()
    pipe = Pipeline([('scaler', StandardScaler()), ('model', cfg['model'])])
    search = RandomizedSearchCV(
        pipe, cfg['params'], n_iter=N_ITER, cv=INNER_CV,
        scoring='r2', random_state=SEED, n_jobs=-1, refit=True
    )
    search.fit(X, y)
    elapsed = round(time.time() - t0, 1)
    tuning_results[name] = {
        'best_r2':     search.best_score_,
        'best_params': {k: (v.item() if hasattr(v, 'item') else v) for k, v in search.best_params_.items()},
        'elapsed_s':   elapsed,
    }
    best_models[name] = search.best_estimator_
    print(f"{name}: R²={search.best_score_:.4f}  ({elapsed}s)")

with open('tuning_results_v2.pkl', 'wb') as f:
    pickle.dump({'tuning': tuning_results, 'models': best_models}, f)

with open('tuning_results_v2.json', 'w') as f:
    json.dump(tuning_results, f, indent=2, default=str)

# seed stability check
best_name = max(tuning_results, key=lambda k: tuning_results[k]['best_r2'])
best_pipe  = best_models[best_name]

print(f"\nSeed stability — {best_name}")
seed_scores = {}
for seed in [42, 123, 456, 789, 2024]:
    cv_s   = RepeatedKFold(n_splits=10, n_repeats=10, random_state=seed)
    scores = cross_val_score(best_pipe, X, y, cv=cv_s, scoring='r2', n_jobs=-1)
    seed_scores[seed] = {'mean': float(scores.mean()), 'std': float(scores.std())}
    print(f"  seed {seed}: R²={scores.mean():.4f} ± {scores.std():.4f}")

means = [v['mean'] for v in seed_scores.values()]
print(f"  cross-seed std: {np.std(means):.4f}")

with open('seed_stability_results.json', 'w') as f:
    json.dump({'model': best_name, 'seeds': seed_scores}, f, indent=2)
