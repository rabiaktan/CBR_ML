import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
from sklearn.base import clone
from sklearn.metrics import make_scorer
import pickle
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('cbr_clean_data.csv')
X  = df.drop('CBR_soaked', axis=1)
y  = df['CBR_soaked']
feats    = X.columns.tolist()
y_binned = pd.qcut(y, q=5, labels=False, duplicates='drop')

with open('tuning_results_v2.pkl', 'rb') as f:
    data = pickle.load(f)
best_models = data['models']

final_results   = pd.read_csv('final_model_results.csv', index_col=0)
best_name = final_results['R2_mean'].idxmax()
best_model = best_models[best_name]
print(f"Ablation with: {best_name} (R²={final_results.loc[best_name,'R2_mean']:.4f})")

rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
cv_splits = list(rkf.split(X, y_binned))

def mape(y_true, y_pred):
    y_true = np.where(np.array(y_true)==0, 1e-10, np.array(y_true))
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

scoring = {
    'r2': 'r2',
    'neg_rmse': 'neg_root_mean_squared_error',
    'neg_mae':  'neg_mean_absolute_error',
    'neg_mape': make_scorer(mape, greater_is_better=False),
}

def run_cv(model, Xf, yf):
    return cross_validate(clone(model), Xf, yf, cv=cv_splits, scoring=scoring, n_jobs=-1)

fi_df = pd.read_csv('feature_importance_shap.csv', index_col=0)
feats_ranked = fi_df['Average'].sort_values(ascending=False).index.tolist()

scenarios = {
    'Full_Set':               feats,
    'Top_3':                  feats_ranked[:3],
    'Top_5':                  feats_ranked[:5],
    'Top_7':                  feats_ranked[:7],
    'Without_Least':          feats_ranked[:-1],
    'Without_Top_1':          [f for f in feats if f != feats_ranked[0]],
}

feat_results = {}
for name, fs in scenarios.items():
    try:
        cv  = run_cv(best_model, X[fs], y)
        r2  = cv['test_r2']
        rms = -cv['test_neg_rmse']
        feat_results[name] = {
            'n_features': len(fs),
            'R2_mean': r2.mean(), 'R2_std': r2.std(),
            'RMSE_mean': rms.mean(), 'RMSE_std': rms.std(),
            'MAE_mean': -cv['test_neg_mae'].mean(), 'MAE_std': cv['test_neg_mae'].std(),
        }
        print(f"{name} ({len(fs)} feats): R²={r2.mean():.4f} ± {r2.std():.4f}")
    except Exception as e:
        print(f"{name}: {e}")

feat_abl = pd.DataFrame(feat_results).T
feat_abl.to_csv('ablation_feature_sets.csv')

transform_scenarios = {
    'No_Transform':     {'X': False, 'y': False},
    'Log_Target_Only':  {'X': False, 'y': True},
    'Log_Feats_Only':   {'X': True,  'y': False},
    'Log_Both':         {'X': True,  'y': True},
}

trans_results = {}
for name, tr in transform_scenarios.items():
    try:
        Xt = np.log1p(X.clip(lower=0)) if tr['X'] else X.copy()
        yt = np.log(y + max(0, -y.min()) + 1) if tr['y'] else y.copy()
        cv  = run_cv(best_model, Xt, yt)
        r2  = cv['test_r2']
        rms = -cv['test_neg_rmse']
        trans_results[name] = {
            'R2_mean': r2.mean(), 'R2_std': r2.std(),
            'RMSE_mean': rms.mean(), 'RMSE_std': rms.std(),
        }
        print(f"{name}: R²={r2.mean():.4f} ± {r2.std():.4f}")
    except Exception as e:
        print(f"{name}: {e}")

trans_abl = pd.DataFrame(trans_results).T
trans_abl.to_csv('ablation_log_transforms.csv')

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

fa = feat_abl.sort_values('R2_mean', ascending=True)
clrs = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(fa)))
bars = axes[0,0].barh(fa.index, fa['R2_mean'], xerr=fa['R2_std'],
                      color=clrs, edgecolor='black', capsize=4, alpha=0.85)
bars[fa['R2_mean'].argmax()].set_edgecolor('red')
bars[fa['R2_mean'].argmax()].set_linewidth(2.5)
axes[0,0].set_xlabel('R²'); axes[0,0].set_title('(a) Feature Set — R²')
axes[0,0].grid(alpha=0.3, axis='x')

fa2 = feat_abl.sort_values('RMSE_mean', ascending=False)
axes[0,1].barh(fa2.index, fa2['RMSE_mean'], xerr=fa2['RMSE_std'],
               color=plt.cm.RdYlGn_r(np.linspace(0.3,0.9,len(fa2))),
               edgecolor='black', capsize=4, alpha=0.85)
axes[0,1].set_xlabel('RMSE'); axes[0,1].set_title('(b) Feature Set — RMSE')
axes[0,1].grid(alpha=0.3, axis='x')

sc = axes[0,2].scatter(feat_abl['n_features'], feat_abl['R2_mean'],
                        s=180, c=feat_abl['R2_mean'], cmap='RdYlGn', edgecolors='black')
for idx, row in feat_abl.iterrows():
    axes[0,2].annotate(idx, (row['n_features'], row['R2_mean']), xytext=(4,4),
                        textcoords='offset points', fontsize=8)
axes[0,2].set_xlabel('# Features'); axes[0,2].set_ylabel('R²')
axes[0,2].set_title('(c) Feature Count vs R²'); axes[0,2].grid(alpha=0.3)
plt.colorbar(sc, ax=axes[0,2])

clrs4 = ['#5DA5DA','#FAA43A','#60BD68','#F17CB0']
bars2 = axes[1,0].bar(trans_abl.index, trans_abl['R2_mean'], yerr=trans_abl['R2_std'],
                       color=clrs4, edgecolor='black', capsize=5, alpha=0.85)
bars2[trans_abl['R2_mean'].argmax()].set_edgecolor('red')
bars2[trans_abl['R2_mean'].argmax()].set_linewidth(2.5)
axes[1,0].set_ylabel('R²'); axes[1,0].set_title('(d) Log Transform — R²')
axes[1,0].set_xticklabels(trans_abl.index, rotation=30, ha='right')
axes[1,0].grid(alpha=0.3, axis='y')

axes[1,1].bar(trans_abl.index, trans_abl['RMSE_mean'], yerr=trans_abl['RMSE_std'],
               color=clrs4, edgecolor='black', capsize=5, alpha=0.85)
axes[1,1].set_ylabel('RMSE'); axes[1,1].set_title('(e) Log Transform — RMSE')
axes[1,1].set_xticklabels(trans_abl.index, rotation=30, ha='right')
axes[1,1].grid(alpha=0.3, axis='y')

all_idx  = list(feat_abl.index) + list(trans_abl.index)
all_r2   = list(feat_abl['R2_mean']) + list(trans_abl['R2_mean'])
all_rmse = list(feat_abl['RMSE_mean']) + list(trans_abl['RMSE_mean'])
hm = pd.DataFrame({
    'R² (norm)':   (np.array(all_r2)-min(all_r2))/(max(all_r2)-min(all_r2)),
    'RMSE (norm)': 1-(np.array(all_rmse)-min(all_rmse))/(max(all_rmse)-min(all_rmse)),
}, index=all_idx)
sns.heatmap(hm.T, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[1,2])
axes[1,2].set_title('(f) Summary (normalized)')
axes[1,2].set_xticklabels(axes[1,2].get_xticklabels(), rotation=40, ha='right')

plt.tight_layout()
plt.savefig('16_ablation_study.png', dpi=300, bbox_inches='tight')
plt.close()
