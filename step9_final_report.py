import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

final   = pd.read_csv('final_model_results.csv', index_col=0)
friedman = pd.read_csv('friedman_test_results.csv', index_col=0)
ranks   = pd.read_csv('friedman_avg_ranks.csv')
noise   = pd.read_csv('noise_robustness_results.csv')
feat_abl = pd.read_csv('ablation_feature_sets.csv', index_col=0)
fi      = pd.read_csv('feature_importance_shap.csv', index_col=0)

print("=" * 60)
print("CBR ML PIPELINE — FINAL SUMMARY")
print("=" * 60)

print("\nModel Performance (outer CV, 100 iter):")
print(final[['R2_mean','R2_std','RMSE_mean','RMSE_std','MAE_mean','MAE_std']].round(4).to_string())

best = final['R2_mean'].idxmax()
print(f"\nBest model: {best}")
print(f"  R²   = {final.loc[best,'R2_mean']:.4f} ± {final.loc[best,'R2_std']:.4f}")
print(f"  RMSE = {final.loc[best,'RMSE_mean']:.4f} ± {final.loc[best,'RMSE_std']:.4f}")
print(f"  MAE  = {final.loc[best,'MAE_mean']:.4f} ± {final.loc[best,'MAE_std']:.4f}")

print("\nFriedman test:")
for metric in friedman.index:
    row = friedman.loc[metric]
    print(f"  {metric}: chi2={float(row['statistic']):.2f}, p={float(row['p_value']):.2e}, sig={row['significant']}")

print("\nFriedman average ranks:")
print(ranks.sort_values('Avg_Rank').to_string(index=False))

print("\nTop feature importances (SHAP avg %):")
print(fi['Average'].sort_values(ascending=False).round(2).to_string())

et20 = noise[noise['Model']==best]
if not et20.empty:
    base = et20[et20['Noise_Level']==0.0]['R2_mean'].values[0]
    at20 = et20[et20['Noise_Level']==0.20]['R2_mean'].values[0]
    print(f"\nNoise robustness ({best}): baseline={base:.4f}, @20%={at20:.4f}, "
          f"degradation={(base-at20)/base*100:.1f}%")

print("\nAblation (feature sets):")
print(feat_abl[['n_features','R2_mean','R2_std']].round(4).sort_values('R2_mean', ascending=False).to_string())
