import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

# 1. Load data from Part 1
df = pd.read_csv('appointments_clean.csv')

# Keep appointment_id aside for Task 3.5(e)
appointment_ids = df['appointment_id'].copy()

# Task 3.1: Features and split
X = df[['age', 'distance_km', 'fee_inr', 'lead_days', 'prior_no_shows', 'reminder_sent', 'booking_channel', 'visit_type']].copy()
X['reminder_sent'] = X['reminder_sent'].map({'Yes': 1, 'No': 0})
X = pd.get_dummies(X, columns=['booking_channel', 'visit_type'], drop_first=False)
y = df['no_show']

# Do not shuffle or re-sort rows after loading appointments_clean.csv
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"--- Task 3.1 Results ---")
print(f"Training size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
print(f"No-shows in test set: {y_test.sum()}")
baseline_acc = (y_test == 0).mean()
print(f"Baseline Accuracy (always predicting 0/Attended): {baseline_acc:.4f}\n")


# Task 3.2: Logistic Regression & Coefficients
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_s, y_train)
y_pred_lr = lr.predict(X_test_s)
y_prob_lr = lr.predict_proba(X_test_s)[:, 1]

print("--- Task 3.2: Logistic Regression Results ---")
print("Accuracy:", round(accuracy_score(y_test, y_pred_lr), 3))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("Precision:", round(precision_score(y_test, y_pred_lr), 3))
print("Recall:", round(recall_score(y_test, y_pred_lr), 3))
print("F1 Score:", round(f1_score(y_test, y_pred_lr), 3))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob_lr), 3))

# Reminder coefficient analysis
reminder_idx = list(X.columns).index('reminder_sent')
coef_rem = lr.coef_[0][reminder_idx]
scale_rem = scaler.scale_[reminder_idx]
print(f"Reminder Coef: {coef_rem:.3f}")
print(f"np.exp(coef): {np.exp(coef_rem):.3f}")
print(f"np.exp(coef / scale) [Real No->Yes change odds]: {np.exp(coef_rem / scale_rem):.3f}\n")


# Task 3.3: Decision Tree & Overfitting
dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train, y_train)

dt_depth3 = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_depth3.fit(X_train, y_train)

print("--- Task 3.3: Decision Tree Results ---")
print(f"Full-depth Train Acc: {accuracy_score(y_train, dt_full.predict(X_train)):.2f}, Test Acc: {accuracy_score(y_test, dt_full.predict(X_test)):.2f}")
print(f"Depth-3 Train Acc: {accuracy_score(y_train, dt_depth3.predict(X_train)):.2f}, Test Acc: {accuracy_score(y_test, dt_depth3.predict(X_test)):.2f}\n")


# Task 3.4: Random Forest & Comparison
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

print("--- Task 3.4: Random Forest Results ---")
print("Accuracy:", round(accuracy_score(y_test, y_pred_rf), 3))
print("Recall:", round(recall_score(y_test, y_pred_rf), 3))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob_rf), 3))

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Top 3 Feature Importances:\n", importances.head(3), "\n")


# Task 3.5: Tuning, Cross-Validation & Threshold Selection
lr_pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])

cv_auc_lr = cross_val_score(lr_pipe, X_train, y_train, cv=5, scoring="roc_auc").mean()
cv_auc_dt3 = cross_val_score(DecisionTreeClassifier(max_depth=3, random_state=42), X_train, y_train, cv=5, scoring="roc_auc").mean()
cv_auc_rf = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42), X_train, y_train, cv=5, scoring="roc_auc").mean()

print("--- Task 3.5: Cross-Validated Training ROC-AUC ---")
print(f"Logistic Regression CV AUC: {cv_auc_lr:.3f}")
print(f"Depth-3 Tree CV AUC: {cv_auc_dt3:.3f}")
print(f"Random Forest CV AUC: {cv_auc_rf:.3f}")

grid = GridSearchCV(RandomForestClassifier(random_state=42), {"n_estimators": [100, 200], "max_depth": [3, 5, 8]}, cv=5, scoring="roc_auc")
grid.fit(X_train, y_train)
print("GridSearchCV Best Params:", grid.best_params_)
print("GridSearchCV Best Score (ROC-AUC):", round(grid.best_score_, 3))

# Out-of-fold training probabilities for threshold comparison
oof_probs = cross_val_predict(lr_pipe, X_train, y_train, cv=5, method="predict_proba")[:, 1]

print("\n--- Out-of-Fold Threshold Comparison (0.5 vs 0.35) ---")
for thresh in [0.5, 0.35]:
    preds = (oof_probs >= thresh).astype(int)
    print(f"Threshold {thresh} -> Precision: {precision_score(y_train, preds):.2f}, Recall: {recall_score(y_train, preds):.2f}, F1: {f1_score(y_train, preds):.2f}, Flagged: {preds.sum()}")

# Final Evaluation on Test Set at threshold 0.35
lr_pipe.fit(X_train, y_train)
test_probs = lr_pipe.predict_proba(X_test)[:, 1]
final_preds = (test_probs >= 0.35).astype(int)

print("\n--- Final Test Set Evaluation (Logistic Regression @ Threshold 0.35) ---")
print("Precision:", round(precision_score(y_test, final_preds), 3))
print("Recall:", round(recall_score(y_test, final_preds), 3))
print("Flagged Count:", final_preds.sum())
print("ROC-AUC:", round(roc_auc_score(y_test, test_probs), 3))

# Save test_risk_scores.csv
test_results = pd.DataFrame({
    'appointment_id': appointment_ids.loc[X_test.index],
    'no_show': y_test,
    'risk_prob': test_probs,
    'flag': final_preds
})
test_results.to_csv('test_risk_scores.csv', index=False)
print("\n'test_risk_scores.csv' has been successfully generated and saved!")