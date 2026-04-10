import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import shap
import pickle
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('cbr_clean_data.csv')
X  = df.drop('CBR_soaked', axis=1).reset_index(drop=True)
y  = df['CBR_soaked']
feats = X.columns.tolist()

with open('tuning_results_v2.pkl', 'rb') as f:
    data = pickle.load(f)
best_models = data['models']

final_results = pd.read_csv('final_model_results.csv', index_col=0)
top3 = final_results.nlargest(3, 'R2_mean').index.tolist()
print(f"SHAP analysis for: {top3}")

TREE_MODELS   = {'ExtraTrees', 'RandomForest', 'XGBoost', 'LightGBM', 'CatBoost'}
KERNEL_MODELS = {'SVR', 'GPR', 'ANN', 'ElasticNet'}
BG_SIZE = 100

def unwrap(model, X_df):
    if isinstance(model, Pipeline) and 'model' in model.named_steps:
        core = model.named_steps['model']
        X_sc = pd.DataFrame(model.named_steps['scaler'].transform(X_df), columns=feats) \
               if 'scaler' in model.named_steps else X_df
        return core, X_sc
    return model, X_df

shap_vals = {}
explainers = {}
X_used = {}

for name in top3:
    core, X_sc = unwrap(best_models[name], X)
    X_used[name] = X_sc
    try:
        if name in TREE_MODELS:
            expl = shap.TreeExplainer(core)
            sv   = expl.shap_values(X_sc)
        else:
            np.random.seed(42)
            bg   = shap.sample(X_sc, BG_SIZE)
            expl = shap.KernelExplainer(core.predict, bg)
            sv   = expl.shap_values(X_sc[:BG_SIZE])
        shap_vals[name]  = sv
        explainers[name] = expl
        print(f"{name}: {'TreeSHAP' if name in TREE_MODELS else 'KernelSHAP'} done")
    except Exception as e:
        print(f"{name}: {e}")

for name in top3:
    if name not in shap_vals:
        continue
    sv  = shap_vals[name]
    X_p = X_used[name][:len(sv)]

    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, X_p, feature_names=feats, show=False)
    plt.title(f'SHAP Summary — {name}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'07_shap_summary_{name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(9, 6))
    shap.summary_plot(sv, X_p, feature_names=feats, plot_type='bar', show=False)
    plt.title(f'SHAP Feature Importance — {name}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'08_shap_importance_{name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()

# feature importance table
fi = {}
for name in top3:
    if name in shap_vals:
        fi[name] = np.abs(shap_vals[name]).mean(axis=0)

fi_df = pd.DataFrame(fi, index=feats)
fi_df = fi_df.apply(lambda c: c / c.sum() * 100)
fi_df['Average'] = fi_df.mean(axis=1)
fi_df = fi_df.sort_values('Average', ascending=False)
fi_df.to_csv('feature_importance_shap.csv')
print(fi_df.round(2))

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fi_df[top3].plot(kind='barh', ax=axes[0])
axes[0].set_xlabel('Importance (%)'); axes[0].set_title('SHAP — Top 3 Models')
axes[0].grid(alpha=0.3, axis='x')

colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(fi_df)))
fi_df['Average'].plot(kind='barh', ax=axes[1], color=colors, edgecolor='black')
axes[1].set_xlabel('Average Importance (%)'); axes[1].set_title('Average SHAP Importance')
axes[1].grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('10_feature_importance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# dependence plots for top-4 features
best_name = top3[0]
if best_name in shap_vals:
    sv_b = shap_vals[best_name]
    X_b  = X_used[best_name][:len(sv_b)]
    top4 = fi_df.index[:4].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.ravel()
    for i, feat in enumerate(top4):
        plt.sca(axes[i])
        shap.dependence_plot(feats.index(feat), sv_b, X_b, feature_names=feats, show=False, ax=axes[i])
        axes[i].set_title(f'SHAP Dependence: {feat}', fontweight='bold')
    plt.suptitle(f'SHAP Dependence — {best_name}', fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('12_shap_dependence_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
