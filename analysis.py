"""
Student Performance Prediction
Data Analysis and Predictive Modeling Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import warnings
import os
import pickle

warnings.filterwarnings('ignore')
np.random.seed(42)

os.makedirs('./outputs', exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1.  GENERATE REALISTIC SYNTHETIC DATASET
# ─────────────────────────────────────────────────────────────
n = 395

school      = np.random.choice(['GP', 'MS'], n, p=[0.75, 0.25])
sex         = np.random.choice(['M', 'F'], n, p=[0.47, 0.53])
age         = np.random.randint(15, 23, n)
address     = np.random.choice(['U', 'R'], n, p=[0.70, 0.30])
famsize     = np.random.choice(['LE3', 'GT3'], n, p=[0.35, 0.65])
Pstatus     = np.random.choice(['T', 'A'], n, p=[0.85, 0.15])
Medu        = np.random.randint(0, 5, n)
Fedu        = np.random.randint(0, 5, n)
studytime   = np.random.randint(1, 5, n)     # 1=<2h, 2=2-5h, 3=5-10h, 4=>10h
failures    = np.random.choice([0, 1, 2, 3], n, p=[0.67, 0.20, 0.08, 0.05])
schoolsup   = np.random.choice(['yes', 'no'], n, p=[0.30, 0.70])
famsup      = np.random.choice(['yes', 'no'], n, p=[0.55, 0.45])
paid        = np.random.choice(['yes', 'no'], n, p=[0.35, 0.65])
activities  = np.random.choice(['yes', 'no'], n, p=[0.50, 0.50])
nursery     = np.random.choice(['yes', 'no'], n, p=[0.75, 0.25])
higher      = np.random.choice(['yes', 'no'], n, p=[0.93, 0.07])
internet    = np.random.choice(['yes', 'no'], n, p=[0.80, 0.20])
romantic    = np.random.choice(['yes', 'no'], n, p=[0.33, 0.67])
famrel      = np.random.randint(1, 6, n)
freetime    = np.random.randint(1, 6, n)
goout       = np.random.randint(1, 6, n)
Dalc        = np.random.randint(1, 6, n)
Walc        = np.random.randint(1, 6, n)
health      = np.random.randint(1, 6, n)
absences    = np.clip(np.random.negative_binomial(2, 0.35, n), 0, 75)

# ── Generate G1 with realistic drivers
G1_base = (
    10
    + 1.5 * studytime
    - 2.5 * failures
    - 0.06 * absences
    + 0.5 * Medu
    + 0.3 * Fedu
    + 1.0 * (higher == 'yes').astype(int)
    - 0.5 * goout
    - 0.3 * Dalc
    + np.random.normal(0, 2.2, n)
)
G1 = np.clip(np.round(G1_base).astype(int), 0, 20)

G2_base = G1 + np.random.normal(0.1, 1.5, n)
G2 = np.clip(np.round(G2_base).astype(int), 0, 20)

# G3: mostly determined by G1/G2 but some real failures (grade = 0)
fail_mask = np.random.random(n) < 0.05
G3_base = 0.5 * G1 + 0.5 * G2 + np.random.normal(0, 1.3, n)
G3 = np.clip(np.round(G3_base).astype(int), 0, 20)
G3[fail_mask] = 0

df = pd.DataFrame({
    'school': school, 'sex': sex, 'age': age, 'address': address,
    'famsize': famsize, 'Pstatus': Pstatus, 'Medu': Medu, 'Fedu': Fedu,
    'studytime': studytime, 'failures': failures,
    'schoolsup': schoolsup, 'famsup': famsup, 'paid': paid,
    'activities': activities, 'nursery': nursery, 'higher': higher,
    'internet': internet, 'romantic': romantic,
    'famrel': famrel, 'freetime': freetime, 'goout': goout,
    'Dalc': Dalc, 'Walc': Walc, 'health': health,
    'absences': absences, 'G1': G1, 'G2': G2, 'G3': G3
})

df.to_csv('./outputs/student_dataset_raw.csv', index=False)
print("Dataset shape:", df.shape)
print(df.describe())

# ─────────────────────────────────────────────────────────────
# 2.  DATA CLEANING & PREPARATION
# ─────────────────────────────────────────────────────────────
print("\n=== Missing Values ===")
print(df.isnull().sum())
print("\n=== Duplicates:", df.duplicated().sum())

# Encode binary categoricals
binary_cols = ['school','sex','address','famsize','Pstatus',
               'schoolsup','famsup','paid','activities','nursery',
               'higher','internet','romantic']
le = LabelEncoder()
df_encoded = df.copy()
for col in binary_cols:
    df_encoded[col] = le.fit_transform(df[col])

# Add pass/fail column (pass = G3 >= 10)
df['pass_fail'] = (df['G3'] >= 10).astype(int)
df_encoded['pass_fail'] = df['pass_fail']

df_encoded.to_csv('./outputs/student_dataset_cleaned.csv', index=False)
print("\nCleaned dataset saved.")

# ─────────────────────────────────────────────────────────────
# 3.  DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────────────────────
stats = df[['G1','G2','G3','studytime','failures','absences','age']].describe().round(2)
print("\n=== Descriptive Statistics ===")
print(stats)

pass_rate = df['pass_fail'].mean() * 100
print(f"\nOverall pass rate: {pass_rate:.1f}%")
print(f"Mean final grade (G3): {df['G3'].mean():.2f}")

# ─────────────────────────────────────────────────────────────
# 4.  VISUALIZATIONS
# ─────────────────────────────────────────────────────────────
PALETTE = ['#065A82', '#1C7293', '#028090', '#00A896', '#02C39A', '#F0F3BD']
DARK    = '#065A82'
ACCENT  = '#F96167'

# ── Fig 1: Grade distributions
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle('Distribution of Student Grades', fontsize=14, fontweight='bold', y=1.01)
for i, (col, ax) in enumerate(zip(['G1','G2','G3'], axes)):
    ax.hist(df[col], bins=20, color=PALETTE[i], edgecolor='white', linewidth=0.5)
    ax.axvline(df[col].mean(), color=ACCENT, linestyle='--', linewidth=1.5, label=f'Mean={df[col].mean():.1f}')
    ax.set_title(col, fontsize=12, fontweight='bold')
    ax.set_xlabel('Grade (0–20)')
    ax.set_ylabel('Count')
    ax.legend(fontsize=9)
    ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig('./outputs/fig1_grade_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Fig 2: Correlation heatmap
fig, ax = plt.subplots(figsize=(11, 8))
num_cols = ['age','Medu','Fedu','studytime','failures','famrel',
            'freetime','goout','Dalc','Walc','health','absences','G1','G2','G3']
corr = df_encoded[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='Blues', annot=True, fmt='.2f',
            annot_kws={'size': 7}, ax=ax, square=True,
            linewidths=0.5, cbar_kws={'shrink': 0.8})
ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('./outputs/fig2_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Fig 3: Key relationships with G3
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('Key Factors Affecting Final Grade (G3)', fontsize=14, fontweight='bold')

# G1 vs G3
axes[0,0].scatter(df['G1'], df['G3'], alpha=0.5, color=DARK, s=20)
m, b = np.polyfit(df['G1'], df['G3'], 1)
x_line = np.linspace(0, 20, 100)
axes[0,0].plot(x_line, m*x_line+b, color=ACCENT, linewidth=2)
axes[0,0].set_xlabel('First Period Grade (G1)')
axes[0,0].set_ylabel('Final Grade (G3)')
axes[0,0].set_title('G1 vs G3')
axes[0,0].spines[['top','right']].set_visible(False)

# Studytime vs G3 (box)
study_labels = ['< 2h', '2–5h', '5–10h', '> 10h']
study_groups = [df[df['studytime']==i]['G3'].values for i in [1,2,3,4]]
bp = axes[0,1].boxplot(study_groups, patch_artist=True, labels=study_labels,
                        medianprops=dict(color=ACCENT, linewidth=2))
for patch, c in zip(bp['boxes'], PALETTE[:4]):
    patch.set_facecolor(c); patch.set_alpha(0.7)
axes[0,1].set_xlabel('Weekly Study Time')
axes[0,1].set_ylabel('Final Grade (G3)')
axes[0,1].set_title('Study Time vs G3')
axes[0,1].spines[['top','right']].set_visible(False)

# Failures vs G3
fail_groups = [df[df['failures']==i]['G3'].values for i in [0,1,2,3]]
bp2 = axes[1,0].boxplot(fail_groups, patch_artist=True, labels=['0','1','2','3'],
                         medianprops=dict(color=ACCENT, linewidth=2))
for patch, c in zip(bp2['boxes'], PALETTE[:4]):
    patch.set_facecolor(c); patch.set_alpha(0.7)
axes[1,0].set_xlabel('Past Failures')
axes[1,0].set_ylabel('Final Grade (G3)')
axes[1,0].set_title('Past Failures vs G3')
axes[1,0].spines[['top','right']].set_visible(False)

# Absences vs G3
axes[1,1].scatter(df['absences'], df['G3'], alpha=0.4, color=DARK, s=20)
m2, b2 = np.polyfit(df['absences'], df['G3'], 1)
x2 = np.linspace(0, df['absences'].max(), 100)
axes[1,1].plot(x2, m2*x2+b2, color=ACCENT, linewidth=2)
axes[1,1].set_xlabel('Number of Absences')
axes[1,1].set_ylabel('Final Grade (G3)')
axes[1,1].set_title('Absences vs G3')
axes[1,1].spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig('./outputs/fig3_key_relationships.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Fig 4: Pass/fail by gender and school
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle('Pass Rate Analysis', fontsize=14, fontweight='bold')

pass_by_sex = df.groupby('sex')['pass_fail'].mean() * 100
axes[0].bar(pass_by_sex.index, pass_by_sex.values,
            color=[DARK, '#1C7293'], edgecolor='white')
axes[0].set_ylim(0, 100)
axes[0].set_ylabel('Pass Rate (%)')
axes[0].set_title('By Gender')
for i, v in enumerate(pass_by_sex.values):
    axes[0].text(i, v+1.5, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
axes[0].spines[['top','right']].set_visible(False)

pass_by_school = df.groupby('school')['pass_fail'].mean() * 100
axes[1].bar(pass_by_school.index, pass_by_school.values,
            color=[DARK, '#028090'], edgecolor='white')
axes[1].set_ylim(0, 100)
axes[1].set_ylabel('Pass Rate (%)')
axes[1].set_title('By School')
for i, v in enumerate(pass_by_school.values):
    axes[1].text(i, v+1.5, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
axes[1].spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig('./outputs/fig4_pass_rate_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Fig 5: Feature importance (correlation with G3)
corr_g3 = df_encoded[num_cols].corr()['G3'].drop('G3').sort_values()
colors_bar = [ACCENT if v < 0 else DARK for v in corr_g3.values]
fig, ax = plt.subplots(figsize=(9, 6))
corr_g3.plot(kind='barh', ax=ax, color=colors_bar)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Feature Correlation with Final Grade (G3)', fontsize=13, fontweight='bold')
ax.set_xlabel('Pearson Correlation Coefficient')
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig('./outputs/fig5_feature_correlation.png', dpi=150, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────────────────────
# 5.  MODEL 1 — LINEAR REGRESSION (predict G3)
# ─────────────────────────────────────────────────────────────
features_reg = ['G1','G2','studytime','failures','absences',
                'Medu','Fedu','Dalc','Walc','goout','age',
                'schoolsup','famsup','internet','higher']

X_reg = df_encoded[features_reg]
y_reg = df_encoded['G3']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42)

scaler_reg = StandardScaler()
X_train_rs = scaler_reg.fit_transform(X_train_r)
X_test_rs  = scaler_reg.transform(X_test_r)

lr = LinearRegression()
lr.fit(X_train_rs, y_train_r)
y_pred_r = lr.predict(X_test_rs)

mse  = mean_squared_error(y_test_r, y_pred_r)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test_r, y_pred_r)
r2   = r2_score(y_test_r, y_pred_r)

print("\n=== Linear Regression Results ===")
print(f"RMSE : {rmse:.3f}")
print(f"MAE  : {mae:.3f}")
print(f"R²   : {r2:.3f}")

# ── Save regression model
with open('./outputs/linear_regression_model.pkl', 'wb') as f:
    pickle.dump({'model': lr, 'scaler': scaler_reg, 'features': features_reg}, f)

# ── Regression visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Linear Regression — Model Evaluation', fontsize=13, fontweight='bold')

axes[0].scatter(y_test_r, y_pred_r, alpha=0.6, color=DARK, s=30)
mn, mx = y_test_r.min()-0.5, y_test_r.max()+0.5
axes[0].plot([mn, mx], [mn, mx], color=ACCENT, linewidth=2, label='Perfect fit')
axes[0].set_xlabel('Actual Grade (G3)')
axes[0].set_ylabel('Predicted Grade')
axes[0].set_title(f'Actual vs Predicted\nR² = {r2:.3f}')
axes[0].legend()
axes[0].spines[['top','right']].set_visible(False)

residuals = y_test_r - y_pred_r
axes[1].hist(residuals, bins=20, color=DARK, edgecolor='white')
axes[1].axvline(0, color=ACCENT, linestyle='--', linewidth=2)
axes[1].set_xlabel('Residual (Actual − Predicted)')
axes[1].set_ylabel('Count')
axes[1].set_title('Residual Distribution')
axes[1].spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig('./outputs/fig6_regression_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Coefficients plot
coef_df = pd.DataFrame({'Feature': features_reg, 'Coefficient': lr.coef_}).sort_values('Coefficient')
fig, ax = plt.subplots(figsize=(9, 5))
colors_c = [ACCENT if c < 0 else DARK for c in coef_df['Coefficient']]
ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors_c)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Linear Regression Coefficients (Standardized)', fontsize=12, fontweight='bold')
ax.set_xlabel('Coefficient Value')
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig('./outputs/fig7_regression_coefficients.png', dpi=150, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────────────────────
# 6.  MODEL 2 — LOGISTIC REGRESSION (pass/fail)
# ─────────────────────────────────────────────────────────────
features_cls = ['G1','G2','studytime','failures','absences',
                'Medu','Fedu','Dalc','Walc','goout','age',
                'schoolsup','famsup','internet','higher']

X_cls = df_encoded[features_cls]
y_cls = df_encoded['pass_fail']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls)

scaler_cls = StandardScaler()
X_train_cs = scaler_cls.fit_transform(X_train_c)
X_test_cs  = scaler_cls.transform(X_test_c)

log_reg = LogisticRegression(max_iter=500, random_state=42)
log_reg.fit(X_train_cs, y_train_c)
y_pred_c = log_reg.predict(X_test_cs)

acc  = accuracy_score(y_test_c, y_pred_c)
prec = precision_score(y_test_c, y_pred_c)
rec  = recall_score(y_test_c, y_pred_c)
f1   = f1_score(y_test_c, y_pred_c)
cm   = confusion_matrix(y_test_c, y_pred_c)

print("\n=== Logistic Regression Results ===")
print(f"Accuracy  : {acc:.3f}")
print(f"Precision : {prec:.3f}")
print(f"Recall    : {rec:.3f}")
print(f"F1 Score  : {f1:.3f}")
print(classification_report(y_test_c, y_pred_c, target_names=['Fail','Pass']))

# ── MODEL 3 — DECISION TREE
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train_cs, y_train_c)
y_pred_dt = dt.predict(X_test_cs)

acc_dt  = accuracy_score(y_test_c, y_pred_dt)
prec_dt = precision_score(y_test_c, y_pred_dt)
rec_dt  = recall_score(y_test_c, y_pred_dt)
f1_dt   = f1_score(y_test_c, y_pred_dt)

print("\n=== Decision Tree Results ===")
print(f"Accuracy  : {acc_dt:.3f}")
print(f"Precision : {prec_dt:.3f}")
print(f"Recall    : {rec_dt:.3f}")
print(f"F1 Score  : {f1_dt:.3f}")

# ── Save classification models
with open('./outputs/logistic_regression_model.pkl', 'wb') as f:
    pickle.dump({'model': log_reg, 'scaler': scaler_cls, 'features': features_cls}, f)
with open('./outputs/decision_tree_model.pkl', 'wb') as f:
    pickle.dump({'model': dt, 'scaler': scaler_cls, 'features': features_cls}, f)

# ── Confusion matrix plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle('Classification Model Evaluation', fontsize=13, fontweight='bold')

for ax, (cm_data, title) in zip(axes, [
    (cm, 'Logistic Regression'),
    (confusion_matrix(y_test_c, y_pred_dt), 'Decision Tree')
]):
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Fail','Pass'], yticklabels=['Fail','Pass'],
                cbar=False, annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)

plt.tight_layout()
plt.savefig('./outputs/fig8_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Model comparison bar chart
fig, ax = plt.subplots(figsize=(9, 4))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
log_vals = [acc, prec, rec, f1]
dt_vals  = [acc_dt, prec_dt, rec_dt, f1_dt]
x = np.arange(len(metrics))
w = 0.35
ax.bar(x - w/2, log_vals, w, label='Logistic Regression', color=DARK)
ax.bar(x + w/2, dt_vals,  w, label='Decision Tree',       color='#028090')
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.1)
ax.set_ylabel('Score')
ax.set_title('Model Comparison — Classification Metrics', fontsize=12, fontweight='bold')
ax.legend()
for i, (lv, dv) in enumerate(zip(log_vals, dt_vals)):
    ax.text(i-w/2, lv+0.02, f'{lv:.2f}', ha='center', fontsize=9)
    ax.text(i+w/2, dv+0.02, f'{dv:.2f}', ha='center', fontsize=9)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig('./outputs/fig9_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✅ All visualizations and models saved!")

# ─────────────────────────────────────────────────────────────
# SUMMARY FOR REPORT
# ─────────────────────────────────────────────────────────────
print("\n===== PROJECT SUMMARY =====")
print(f"Dataset: {n} students, {df.shape[1]} features")
print(f"Pass rate: {pass_rate:.1f}%")
print(f"Mean G3: {df['G3'].mean():.2f} ± {df['G3'].std():.2f}")
print(f"\nLinear Regression: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}")
print(f"Logistic Regression: Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}")
print(f"Decision Tree:       Acc={acc_dt:.3f}, Prec={prec_dt:.3f}, Rec={rec_dt:.3f}, F1={f1_dt:.3f}")
print("\nBest Classification Model: Logistic Regression" if acc >= acc_dt else "\nBest Classification Model: Decision Tree")
