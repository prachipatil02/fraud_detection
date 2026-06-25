# ==================== IMPORTS ====================
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("MODEL TRAINING - IEEE-CIS FRAUD DETECTION")
print("=" * 70)

# ==================== LOAD DATA ====================
print("\n📂 Loading processed data...")
X_train = pd.read_csv(r'C:\Users\prachi.patil\Downloads\fraud_detection\data\processed\X_train.csv')
y_train = pd.read_csv(r'C:\Users\prachi.patil\Downloads\fraud_detection\data\processed\y_train.csv').values.ravel()

print(f" X_train shape: {X_train.shape}")
print(f" y_train shape: {y_train.shape}")
print(f" Fraud rate: {y_train.mean() * 100:.2f}%")

# ==================== TRAIN-TEST SPLIT ====================
print("\n Splitting data...")
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_train
)

print(f" Train set: {X_tr.shape}")
print(f" Val set: {X_val.shape}")

# ==================== SCALING ====================
print("\n Scaling features...")
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_val_scaled = scaler.transform(X_val)

# ==================== HANDLE IMBALANCE ====================
print("\n Handling class imbalance with SMOTE...")
smote = SMOTE(random_state=42)
X_tr_balanced, y_tr_balanced = smote.fit_resample(X_tr_scaled, y_tr)

print(f" Before SMOTE: {len(y_tr)}")
print(f" After SMOTE: {len(y_tr_balanced)}")

# ==================== TRAIN XGBOOST ====================
print("\n Training XGBoost model...")
xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    eval_metric='logloss',
    scale_pos_weight=len(y_tr_balanced[y_tr_balanced==0]) / len(y_tr_balanced[y_tr_balanced==1])
)

xgb_model.fit(X_tr_balanced, y_tr_balanced, verbose=True)

# ==================== EVALUATE ====================
print("\n Evaluating model...")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

y_pred = xgb_model.predict(X_val_scaled)
y_proba = xgb_model.predict_proba(X_val_scaled)[:, 1]

print(f"Accuracy:  {accuracy_score(y_val, y_pred):.4f}")
print(f"Precision: {precision_score(y_val, y_pred):.4f}")
print(f"Recall:    {recall_score(y_val, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_val, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_val, y_proba):.4f}")


# ==================== EVALUATE (OPTIMIZED THRESHOLD 0.3) ====================
print("\n Evaluation (Threshold = 0.3 - OPTIMIZED):")
print("=" * 70)

y_pred_optimized = (y_proba > 0.3).astype(int)

print(f"\nAccuracy:  {accuracy_score(y_val, y_pred_optimized):.4f}")
print(f"Precision: {precision_score(y_val, y_pred_optimized):.4f}")
print(f"Recall:    {recall_score(y_val, y_pred_optimized):.4f}")
print(f"F1-Score:  {f1_score(y_val, y_pred_optimized):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_val, y_proba):.4f}")

# ==================== DETAILED CLASSIFICATION REPORT ====================
print("\n Detailed Classification Report (Threshold = 0.3):")
print(classification_report(y_val, y_pred_optimized, target_names=['Legitimate', 'Fraud']))

# ==================== SAVE MODEL ====================
print("\n Saving model...")
joblib.dump(xgb_model, r'C:\Users\prachi.patil\Downloads\fraud_detection\data\models\xgb_model.pkl')
joblib.dump(scaler, r'C:\Users\prachi.patil\Downloads\fraud_detection\data\models\scaler.pkl')
joblib.dump(X_train.columns, r'C:\Users\prachi.patil\Downloads\fraud_detection\data\models\feature_names.pkl')

print("✅ Model saved!")

print("\n Training Complete!")
