# CBR Prediction with Interpretable Machine Learning

Code for the paper: *Interpretable Machine Learning for CBR Prediction: Ensemble Methods with SHAP Analysis*

## Requirements

```
pip install scikit-learn xgboost lightgbm catboost shap scikit-posthocs pandas numpy matplotlib seaborn scipy openpyxl
```

## Usage

Run scripts in order. All scripts read/write from the working directory.

```bash
python step1_data_loading.py       # loads CBR_data.xlsx → cbr_clean_data.csv
python step2_model_training.py     # baseline CV for all 9 models
python step3_hyperparameter_tuning.py  # nested CV + seed stability
python step4_final_evaluation.py   # outer loop performance estimates
python step5_statistical_tests.py  # Friedman + Nemenyi + Wilcoxon
python step6_shap_analysis.py      # TreeSHAP / KernelSHAP
python step7_noise_robustness.py   # feature-proportional noise injection
python step8_ablation_study.py     # feature subset + log transform ablation
python step9_final_report.py       # summary printout
python step10_figures_300dpi.py    # publication-ready figures
```

## Data

Place `CBR_data.xlsx` in the working directory before running step1.
Dataset: 236 soil samples, 8 input features (Gravel, Sand, Fines, LL, PL, PI, MDD, OMC), target: soaked CBR (%).

## Models

ElasticNet, SVR, GPR, Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost, ANN

## Output files

| File | Description |
|------|-------------|
| `cbr_clean_data.csv` | Cleaned dataset |
| `tuning_results_v2.pkl` | Best hyperparameters + fitted pipelines |
| `final_model_results.csv` | CV performance (R², RMSE, MAE) |
| `final_detailed_results.pkl` | Per-fold scores for statistical tests |
| `friedman_test_results.csv` | Friedman omnibus test |
| `friedman_avg_ranks.csv` | Average Friedman ranks |
| `nemenyi_posthoc_results.csv` | Nemenyi pairwise p-values |
| `wilcoxon_test_results.csv` | Wilcoxon signed-rank results |
| `feature_importance_shap.csv` | SHAP feature importance (%) |
| `noise_robustness_results.csv` | R² at different noise levels |
| `ablation_feature_sets.csv` | Feature subset ablation |
| `ablation_log_transforms.csv` | Log transform ablation |
| `seed_stability_results.json` | Cross-seed stability analysis |
