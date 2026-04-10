import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold, cross_validate
from sklearn.metrics import make_scorer
import pickle
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('cbr_clean_data.csv')
X  = df.drop('CBR_soaked', axis=1)
y  = df['CBR_soaked']
y_binned = pd.qcut(y, q=5, labels=False, duplicates='drop')

with open('tuning_results_v2.pkl', 'rb') as f:
    data = pickle.load(f)
best_models   = data['models']
tuning_scores = data['tuning']

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = np.where(y_true == 0, 1e-10, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape_scorer = make_scorer(mape, greater_is_better=False)
scoring = {
    'r2':       'r2',
    'neg_rmse': 'neg_root_mean_squared_error',
    'neg_mae':  'neg_mean_absolute_error',
    'neg_mape': mape_scorer,
}

rkf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)
cv_splits = list(rkf.split(X, y_binned))

final_results  = {}
detailed_results = {}

for name, model in best_models.items():
    try:
        cv = cross_validate(model, X, y, cv=cv_splits, scoring=scoring, n_jobs=-1)
        r2   = cv['test_r2']
        rmse = -cv['test_neg_rmse']
        mae  = -cv['test_neg_mae']
        mape_scores = -cv['test_neg_mape']
        final_results[name] = {
            'R2_mean': r2.mean(),   'R2_std': r2.std(),
            'RMSE_mean': rmse.mean(), 'RMSE_std': rmse.std(),
            'MAE_mean': mae.mean(),   'MAE_std': mae.std(),
            'MAPE_mean': mape_scores.mean(), 'MAPE_std': mape_scores.std(),
        }
        detailed_results[name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape_scores}
        print(f"{name}: R²={r2.mean():.4f} ± {r2.std():.4f}  RMSE={rmse.mean():.4f}")
    except Exception as e:
        print(f"{name} fallback: {e}")
        try:
            rkf2 = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
            cv   = cross_validate(model, X, y, cv=rkf2, scoring=scoring, n_jobs=-1)
            r2   = cv['test_r2']
            rmse = -cv['test_neg_rmse']
            mae  = -cv['test_neg_mae']
            mape_scores = -cv['test_neg_mape']
            final_results[name] = {
                'R2_mean': r2.mean(), 'R2_std': r2.std(),
                'RMSE_mean': rmse.mean(), 'RMSE_std': rmse.std(),
                'MAE_mean': mae.mean(), 'MAE_std': mae.std(),
                'MAPE_mean': mape_scores.mean(), 'MAPE_std': mape_scores.std(),
            }
            detailed_results[name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape_scores}
        except Exception as e2:
            print(f"{name} failed entirely: {e2}")

df_res = pd.DataFrame(final_results).T.round(4).sort_values('R2_mean', ascending=False)
print(df_res[['R2_mean','R2_std','RMSE_mean','RMSE_std','MAE_mean','MAE_std']])

df_res.to_csv('final_model_results.csv')
with open('final_detailed_results.pkl', 'wb') as f:
    pickle.dump(detailed_results, f)

# quick plot
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, col, title, color in zip(axes,
    ['R2_mean','RMSE_mean','MAE_mean'],
    ['R² (mean ± std)','RMSE (mean ± std)','MAE (mean ± std)'],
    ['steelblue','coral','lightgreen']):
    ms = df_res.sort_values(col, ascending=(col != 'R2_mean'))
    err_col = col.replace('mean','std')
    ax.barh(ms.index, ms[col], xerr=ms[err_col], color=color, alpha=0.8, capsize=4)
    ax.set_title(title); ax.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('04_final_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
