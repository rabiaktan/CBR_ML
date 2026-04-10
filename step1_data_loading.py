import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df = pd.read_excel('CBR_data.xlsx', header=1)
df = df.drop(['Unnamed: 0'], axis=1)
df.columns = ['No', 'GS', 'Gravel', 'Sand', 'Fines', 'LL', 'PL', 'PI', 'MDD', 'OMC', 'CBR_soaked', 'CBR_unsoaked']

numeric_cols = ['GS', 'Gravel', 'Sand', 'Fines', 'LL', 'PL', 'PI', 'MDD', 'OMC', 'CBR_soaked', 'CBR_unsoaked']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# drop unused columns
df = df.drop(['CBR_unsoaked', 'No'], axis=1)

# GS has too many missing values
if df['GS'].isnull().sum() / len(df) > 0.5:
    df = df.drop(['GS'], axis=1)

df_clean = df.dropna()
print(f"samples: {len(df_clean)}, features: {df_clean.shape[1]-1}")
print(df_clean.describe().round(3))

features = [c for c in df_clean.columns if c != 'CBR_soaked']

# correlation with target
print(df_clean.corr()['CBR_soaked'].sort_values(ascending=False).round(3))

# EDA plots
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.ravel()

axes[0].set_title('Missing values (original)')
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, ax=axes[0])

axes[1].hist(df_clean['CBR_soaked'], bins=30, edgecolor='black', alpha=0.7)
axes[1].axvline(df_clean['CBR_soaked'].mean(), color='r', linestyle='--', label=f"mean={df_clean['CBR_soaked'].mean():.1f}")
axes[1].axvline(df_clean['CBR_soaked'].median(), color='g', linestyle='--', label=f"median={df_clean['CBR_soaked'].median():.1f}")
axes[1].set_xlabel('CBR (%)'); axes[1].legend(); axes[1].set_title('CBR distribution')

stats.probplot(df_clean['CBR_soaked'], dist='norm', plot=axes[2])
axes[2].set_title('Q-Q plot')

for i, feat in enumerate(features[:6], 3):
    axes[i].boxplot(df_clean[feat])
    axes[i].set_title(feat)
    axes[i].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('01_data_exploration.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(11, 9))
sns.heatmap(df_clean.corr(), annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5)
plt.title('Correlation matrix')
plt.tight_layout()
plt.savefig('02_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

df_clean.to_csv('cbr_clean_data.csv', index=False)
print("saved: cbr_clean_data.csv")
