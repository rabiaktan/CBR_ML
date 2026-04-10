import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('cbr_clean_data.csv')
y  = df['CBR_soaked'].values

# ─────────────────────────────────────────────────────────────────────────────
# Empirical CBR equations from the literature
# All applied to this dataset as-is (no re-calibration)
# This tests transferability, not equation quality.
# ─────────────────────────────────────────────────────────────────────────────

def eq_black_1962(df):
    """
    Black (1962): CBR = (10/PI)^1.33
    Valid for cohesive soils with PI > 5.
    Ref: Black, W.P.M. (1962). A method of estimating the California Bearing
         Ratio of cohesive soils from plasticity data. Geotechnique, 12(4), 271-282.
    """
    PI = df['PI'].clip(lower=5)
    return (10 / PI) ** 1.33

def eq_de_graft_johnson_1972(df):
    """
    De Graft-Johnson & Bhatia (1972): log10(CBR) = 2.56 - 1.15*log10(PI)
    Ref: De Graft-Johnson, J.W.S. & Bhatia, H.S. (1972). The engineering
         characteristics of the laterite gravels of Ghana.
         Proc. 5th Regional Conf. Africa Soil Mech., Luanda.
    """
    PI = df['PI'].clip(lower=1)
    log_cbr = 2.56 - 1.15 * np.log10(PI)
    return 10 ** log_cbr

def eq_patel_desai_2010(df):
    """
    Patel & Desai (2010): CBR = 7.122*MDD - 0.889*OMC - 2.580*PI - 5.898
    Ref: Patel, R.S. & Desai, M.D. (2010). CBR predicted by Index Properties
         for Alluvial Soils of South Gujarat. Indian Geotechnical Conference,
         GEOtrendz, 79-82.
    """
    return 7.122*df['MDD'] - 0.889*df['OMC'] - 2.580*df['PI'] - 5.898

def eq_taskiran_2010(df):
    """
    Taskiran (2010): CBR = 62.36 - 0.682*LL - 0.359*PI
    Regression equation for fine-grained soils.
    Ref: Taskiran, T. (2010). Improvement of CBR by the aid of fly ash.
         Construction and Building Materials, 24(8), 1368-1374.
    Note: Equation fitted on Turkish highway subgrade soils.
    """
    return 62.36 - 0.682*df['LL'] - 0.359*df['PI']

def eq_vinod_cletus_2011(df):
    """
    Vinod & Cletus (2011): CBR = 22*MDD - 0.4*LL - 0.47*PL + 0.31*PI - 34.45
    Ref: Vinod, P. & Cletus, B. (2011). Estimating CBR value of cohesive soils
         from plasticity characteristics. ARPN Journal of Engineering and
         Applied Sciences, 6(6), 1-6.
    """
    return 22*df['MDD'] - 0.4*df['LL'] - 0.47*df['PL'] + 0.31*df['PI'] - 34.45

def eq_aashto_classification(df):
    """
    AASHTO (2004) group index-based CBR approximation.
    Group Index GI = 0.2*a + 0.005*a*c + 0.01*b*d, then CBR ≈ f(GI)
    where a = F200 - 35 (clipped 0-40), b = F200 - 15 (clipped 0-40)
          c = LL - 40 (clipped 0-20),  d = PI - 10 (clipped 0-20)
    CBR ≈ 75 / (1 + GI)  (inverse relationship used in AASHTO design)
    Ref: AASHTO (2004). A Policy on Geometric Design of Highways and Streets.
         American Association of State Highway and Transportation Officials.
    """
    F200 = df['Fines']
    LL   = df['LL']
    PI   = df['PI']
    a = np.clip(F200 - 35, 0, 40)
    b = np.clip(F200 - 15, 0, 40)
    c = np.clip(LL  - 40,  0, 20)
    d = np.clip(PI  - 10,  0, 20)
    GI = 0.2*a + 0.005*a*c + 0.01*b*d
    GI = np.clip(GI, 0.1, None)
    return 75 / (1 + GI)

# ─────────────────────────────────────────────────────────────────────────────
# Evaluate all equations
# ─────────────────────────────────────────────────────────────────────────────

equations = {
    'Black (1962)':                 eq_black_1962,
    'De Graft-Johnson (1972)':      eq_de_graft_johnson_1972,
    'Patel & Desai (2010)':         eq_patel_desai_2010,
    'Taskiran (2010)':              eq_taskiran_2010,
    'Vinod & Cletus (2011)':        eq_vinod_cletus_2011,
    'AASHTO GI-based (2004)':       eq_aashto_classification,
}

emp_results = {}
for name, fn in equations.items():
    try:
        y_pred = fn(df).values
        y_pred = np.clip(y_pred, 0.1, None)   # CBR cannot be negative
        r2   = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae  = mean_absolute_error(y, y_pred)
        emp_results[name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'y_pred': y_pred}
        print(f"{name:<30}  R²={r2:+.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}")
    except Exception as e:
        print(f"{name}: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Load ML results and merge
# ─────────────────────────────────────────────────────────────────────────────

ml = pd.read_csv('final_model_results.csv', index_col=0)

emp_df = pd.DataFrame({k: {'R2': v['R2'], 'RMSE': v['RMSE'], 'MAE': v['MAE']}
                        for k, v in emp_results.items()}).T.round(3)
emp_df['Type'] = 'Empirical'
emp_df['R2_std'] = np.nan

ml_sub = ml[['R2_mean','RMSE_mean','MAE_mean','R2_std']].copy()
ml_sub.columns = ['R2','RMSE','MAE','R2_std']
ml_sub['Type'] = 'ML'

all_df = pd.concat([ml_sub, emp_df[['R2','RMSE','MAE','R2_std','Type']]])
all_df = all_df.sort_values('R2', ascending=False)

print("\n\nFull Comparison (ML + Empirical):")
print(all_df[['Type','R2','RMSE','MAE']].round(3).to_string())

# Save
emp_df[['R2','RMSE','MAE']].round(3).to_csv('empirical_equation_results.csv')
all_df.round(3).to_csv('full_comparison_ml_vs_empirical.csv')
print("\nsaved: empirical_equation_results.csv")
print("saved: full_comparison_ml_vs_empirical.csv")

# ─────────────────────────────────────────────────────────────────────────────
# Scatter plots: predicted vs observed for each empirical equation
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 11))
axes = axes.ravel()
lim  = [0, 50]

for i, (name, res) in enumerate(emp_results.items()):
    ax     = axes[i]
    y_pred = np.clip(res['y_pred'], 0, 50)
    r2     = res['R2']
    rmse   = res['RMSE']

    ax.scatter(y, y_pred, alpha=0.55, s=28, color='#3498db',
               edgecolors='white', linewidths=0.4)
    ax.plot(lim, lim, 'r--', linewidth=1.2, label='1:1 line')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Measured CBR (%)', fontweight='bold')
    ax.set_ylabel('Predicted CBR (%)', fontweight='bold')
    ax.set_title(name, fontweight='bold', fontsize=11)
    ax.text(0.04, 0.94, f'R² = {r2:.3f}\nRMSE = {rmse:.2f}',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    ax.grid(alpha=0.25)

plt.suptitle('Empirical Equations: Predicted vs. Measured CBR\n'
             '(applied to the full 236-sample dataset without re-calibration)',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('Figure_EmpiricalScatter_300dpi.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("saved: Figure_EmpiricalScatter_300dpi.png")

# ─────────────────────────────────────────────────────────────────────────────
# Main comparison figure: ML vs Empirical (R² bar chart)
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 7))

metrics = [('R2','R² Score','higher is better'),
           ('RMSE','RMSE (%)','lower is better'),
           ('MAE','MAE (%)','lower is better')]

ml_names  = list(ml_sub.index)
emp_names = list(emp_df.index)

for ax, (metric, ylabel, note) in zip(axes, metrics):
    ml_vals  = all_df.loc[ml_names,  metric].astype(float)
    emp_vals = all_df.loc[emp_names, metric].astype(float)

    # sort descending for R², ascending for errors
    ml_sorted  = ml_vals.sort_values(ascending=(metric != 'R2'))
    emp_sorted = emp_vals.sort_values(ascending=(metric != 'R2'))

    all_sorted = pd.concat([ml_sorted, emp_sorted])
    colors = ['#2980b9' if n in ml_names else '#e74c3c' for n in all_sorted.index]

    bars = ax.barh(all_sorted.index, all_sorted.values,
                   color=colors, edgecolor='white', linewidth=0.6, alpha=0.88)

    # error bars for ML (R² std)
    if metric == 'R2':
        for j, name in enumerate(all_sorted.index):
            if name in ml_names and not pd.isna(ml_sub.loc[name, 'R2_std']):
                std = float(ml_sub.loc[name, 'R2_std'])
                val = float(ml_sorted[name]) if name in ml_sorted else 0
                ax.errorbar(val, j - len(ml_names) + len(all_sorted) - len(ml_sorted) +
                            list(all_sorted.index).index(name),
                            xerr=std, fmt='none', color='black',
                            capsize=3, linewidth=1.0)

    ax.set_xlabel(ylabel, fontweight='bold')
    ax.set_title(f'{ylabel}\n({note})', fontweight='bold')
    ax.grid(alpha=0.25, axis='x')

    # divider line between ML and empirical
    n_ml = len(ml_names)
    ax.axhline(y=n_ml - 0.5, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)
    ax.text(ax.get_xlim()[1]*0.98, n_ml - 0.3, 'ML above / Empirical below',
            ha='right', va='bottom', fontsize=8, color='gray', style='italic')

ml_patch  = mpatches.Patch(color='#2980b9', label='Machine Learning (CV mean ± std)')
emp_patch = mpatches.Patch(color='#e74c3c', label='Empirical Equation (no re-calibration)')

fig.legend(handles=[ml_patch, emp_patch], loc='lower center',
           ncol=2, fontsize=10, frameon=True,
           bbox_to_anchor=(0.5, -0.04))

plt.suptitle('Comparison of ML Models vs. Empirical CBR Equations\n'
             '(n = 236 samples; empirical equations applied out-of-calibration)',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('Figure_ML_vs_Empirical_300dpi.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("saved: Figure_ML_vs_Empirical_300dpi.png")

# ─────────────────────────────────────────────────────────────────────────────
# Summary table for the paper
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("TABLE — ML vs Empirical (paper-ready summary)")
print("="*70)
print(f"\n{'Method':<30} {'Type':<12} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
print("-"*70)
for name, row in all_df.sort_values('R2', ascending=False).iterrows():
    r2_str = f"{row['R2']:+.3f}" if not pd.isna(row['R2']) else "—"
    print(f"{name:<30} {row['Type']:<12} {r2_str:>8} {row['RMSE']:>8.3f} {row['MAE']:>8.3f}")

print("\nNote: Negative R² indicates the equation performs worse than a naive")
print("mean predictor on this geographically diverse dataset — this reflects")
print("the limited transferability of regionally calibrated equations, not")
print("a fundamental flaw in the equations themselves.")
