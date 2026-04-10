import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    from scikit_posthocs import posthoc_nemenyi_friedman
    HAS_NEMENYI = True
except ImportError:
    HAS_NEMENYI = False
    print("install scikit-posthocs for Nemenyi test")

with open('final_detailed_results.pkl', 'rb') as f:
    detailed = pickle.load(f)

models  = list(detailed.keys())
n_mod   = len(models)
n_iter  = len(detailed[models[0]]['R2'])

r2_mat   = np.array([detailed[m]['R2']   for m in models]).T
rmse_mat = np.array([detailed[m]['RMSE'] for m in models]).T
mae_mat  = np.array([detailed[m]['MAE']  for m in models]).T

# Friedman
friedman = {}
for label, mat in [('R2', r2_mat), ('RMSE', rmse_mat), ('MAE', mae_mat)]:
    stat, p = friedmanchisquare(*mat.T)
    friedman[label] = {'statistic': stat, 'p_value': p, 'significant': p < 0.05}
    print(f"Friedman {label}: chi2={stat:.2f}, p={p:.2e}, sig={p<0.05}")

pd.DataFrame(friedman).T.to_csv('friedman_test_results.csv')

# average ranks
ranks = np.apply_along_axis(lambda r: stats.rankdata(-r, method='average'), 1, r2_mat)
avg_ranks = ranks.mean(axis=0)
rank_df = pd.DataFrame({'Model': models, 'Avg_Rank': avg_ranks}).sort_values('Avg_Rank')
print(rank_df.to_string(index=False))
rank_df.to_csv('friedman_avg_ranks.csv', index=False)

# Nemenyi
nemenyi_res = None
cd = None
if HAS_NEMENYI and friedman['R2']['significant']:
    r2_df = pd.DataFrame(r2_mat, columns=models)
    nemenyi_res = posthoc_nemenyi_friedman(r2_df)
    q_alpha = 2.850
    cd = q_alpha * np.sqrt(n_mod * (n_mod + 1) / (6 * n_iter))
    print(f"Nemenyi CD ≈ {cd:.4f}")
    nemenyi_res.to_csv('nemenyi_posthoc_results.csv')
    for i in range(n_mod):
        for j in range(i+1, n_mod):
            if nemenyi_res.iloc[i,j] < 0.05:
                print(f"  sig: {models[i]} vs {models[j]}: p={nemenyi_res.iloc[i,j]:.4f}")

# Wilcoxon — best vs rest
best = models[int(np.argmax([detailed[m]['R2'].mean() for m in models]))]
print(f"\nWilcoxon — {best} vs others:")
wx_rows = []
for m in models:
    if m == best:
        continue
    try:
        stat, p = wilcoxon(detailed[best]['R2'], detailed[m]['R2'], alternative='greater')
        wx_rows.append({'Model_A': best, 'Model_B': m, 'Statistic': stat, 'p_value': p, 'Significant': p < 0.05})
        print(f"  {best} > {m}: p={p:.4f}  {'*' if p<0.05 else ''}")
    except Exception as e:
        print(f"  {m}: {e}")

pd.DataFrame(wx_rows).to_csv('wilcoxon_test_results.csv', index=False)

# plots
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

rd = rank_df.sort_values('Avg_Rank')
axes[0,0].barh(rd['Model'], rd['Avg_Rank'],
               color=['#2ecc71' if i < 4 else '#3498db' for i in range(len(rd))],
               edgecolor='black', alpha=0.85)
axes[0,0].set_xlabel('Average Rank'); axes[0,0].set_title('(a) Friedman Ranks')
axes[0,0].grid(alpha=0.3, axis='x'); axes[0,0].invert_yaxis()

box_data = [detailed[m]['R2'] for m in rank_df['Model']]
bp = axes[0,1].boxplot(box_data, labels=rank_df['Model'], patch_artist=True)
for i, p in enumerate(bp['boxes']):
    p.set_facecolor('#2ecc71' if i < 4 else '#3498db')
    p.set_alpha(0.7)
axes[0,1].set_ylabel('R²'); axes[0,1].set_title(f'(b) R² Distribution ({n_iter} iter)')
axes[0,1].set_xticklabels(rank_df['Model'], rotation=45, ha='right')
axes[0,1].grid(alpha=0.3, axis='y')

if nemenyi_res is not None:
    mask = np.triu(np.ones_like(nemenyi_res, dtype=bool), k=1)
    sns.heatmap(nemenyi_res, mask=mask, annot=True, fmt='.3f',
                cmap='RdYlGn_r', center=0.05, vmin=0, vmax=0.2,
                square=True, ax=axes[1,0])
    axes[1,0].set_title('(c) Nemenyi p-values')

    sig_mat = (nemenyi_res.values < 0.05).astype(int)
    sns.heatmap(pd.DataFrame(sig_mat, index=models, columns=models),
                mask=mask, annot=True, fmt='d', cmap='RdYlGn',
                center=0.5, square=True, ax=axes[1,1])
    axes[1,1].set_title('(d) Significant differences (1=p<0.05)')
else:
    axes[1,0].axis('off'); axes[1,1].axis('off')

plt.tight_layout()
plt.savefig('05_statistical_tests.png', dpi=300, bbox_inches='tight')
plt.close()
