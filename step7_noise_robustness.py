import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
import pickle
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('cbr_clean_data.csv')
X  = df.drop('CBR_soaked', axis=1).values
y  = df['CBR_soaked'].values
feat_names = df.drop('CBR_soaked', axis=1).columns.tolist()
feat_stds  = df.drop('CBR_soaked', axis=1).std().values

with open('tuning_results_v2.pkl', 'rb') as f:
    data = pickle.load(f)
best_models = data['models']

final_results = pd.read_csv('final_model_results.csv', index_col=0)
test_models   = final_results.nlargest(5, 'R2_mean').index.tolist()

NOISE_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20]
N_MC   = 20
SEED   = 42

y_binned = pd.qcut(pd.Series(y), q=5, labels=False, duplicates='drop')
cv  = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
rng = np.random.default_rng(SEED)

noise_results = {}

for name in test_models:
    model = best_models[name]
    level_r2 = {}
    for p in NOISE_LEVELS:
        trial_r2 = []
        for _ in range(N_MC if p > 0 else 1):
            X_noisy = X + rng.normal(0, p * feat_stds, size=X.shape) if p > 0 else X.copy()
            X_df = pd.DataFrame(X_noisy, columns=feat_names)
            scores = cross_val_score(model, X_df, y,
                                     cv=list(cv.split(X_df, y_binned)),
                                     scoring='r2', n_jobs=-1)
            trial_r2.append(scores.mean())
        level_r2[p] = {'mean': np.mean(trial_r2), 'std': np.std(trial_r2)}
        print(f"{name} p={p:.0%}: R²={np.mean(trial_r2):.4f} ± {np.std(trial_r2):.4f}")
    noise_results[name] = level_r2

rows = [{'Model': m, 'Noise_Level': p, 'R2_mean': v['mean'], 'R2_std': v['std']}
        for m, levels in noise_results.items() for p, v in levels.items()]
pd.DataFrame(rows).to_csv('noise_robustness_results.csv', index=False)

# degradation
print("\nDegradation @20% noise:")
for name in test_models:
    b = noise_results[name][0.0]['mean']
    n = noise_results[name][0.20]['mean']
    print(f"  {name}: {(b-n)/abs(b)*100:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(test_models)))

for i, name in enumerate(test_models):
    means = [noise_results[name][p]['mean'] for p in NOISE_LEVELS]
    stds  = [noise_results[name][p]['std']  for p in NOISE_LEVELS]
    xs    = [p*100 for p in NOISE_LEVELS]
    axes[0].plot(xs, means, marker='o', color=colors[i], label=name, linewidth=2)
    axes[0].fill_between(xs, np.array(means)-np.array(stds),
                         np.array(means)+np.array(stds), alpha=0.12, color=colors[i])

axes[0].set_xlabel('Noise Level (% of feature std)')
axes[0].set_ylabel('R²')
axes[0].set_title('(a) R² vs Noise Level')
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

for i, name in enumerate(test_models):
    b = noise_results[name][0.0]['mean']
    deg = [(b - noise_results[name][p]['mean']) / abs(b) * 100 for p in NOISE_LEVELS[1:]]
    axes[1].plot([p*100 for p in NOISE_LEVELS[1:]], deg,
                 marker='s', color=colors[i], label=name, linewidth=2)

axes[1].axhline(y=7, color='red', linestyle='--', linewidth=1.2)
axes[1].set_xlabel('Noise Level (% of feature std)')
axes[1].set_ylabel('R² Degradation (%)')
axes[1].set_title('(b) Performance Degradation')
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('13_noise_robustness.png', dpi=300, bbox_inches='tight')
plt.close()
