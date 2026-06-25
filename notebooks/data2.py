# ==================== IMPORTS ====================
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("DATA PREPROCESSING - IEEE-CIS FRAUD DETECTION")
print("=" * 70)

# ==================== LOAD DATA ====================
print("\n📂 Loading merged datasets...")
train = pd.read_csv(r'C:\Users\prachi.patil\Downloads\fraud_detection\data\processed\train_merged.csv')
test = pd.read_csv(r'C:\Users\prachi.patil\Downloads\fraud_detection\data\processed\test_merged.csv')

print(f" Train shape: {train.shape}")
print(f" Test shape: {test.shape}")

# ==================== HANDLE MISSING VALUES ====================
print("\n Handling missing values...")

# Drop columns with > 50% missing
missing_pct = (train.isnull().sum() / len(train) * 100)
cols_to_drop = missing_pct[missing_pct > 50].index
print(f"Dropping {len(cols_to_drop)} columns with >50% missing values")
train = train.drop(cols_to_drop, axis=1, errors='ignore')
test = test.drop(cols_to_drop, axis=1, errors='ignore')

# Fill remaining missing values
numeric_cols = train.select_dtypes(include=[np.number]).columns
categorical_cols = train.select_dtypes(include=['object']).columns

# Numeric: fill with median
for col in numeric_cols:
    if col != 'isFraud':
        train[col].fillna(train[col].median(), inplace=True)
        if col in test.columns:
            test[col].fillna(train[col].median(), inplace=True)

# Categorical: fill with mode
for col in categorical_cols:
    mode_val = train[col].mode()[0] if len(train[col].mode()) > 0 else 'Unknown'
    train[col].fillna(mode_val, inplace=True)
    if col in test.columns:
        test[col].fillna(mode_val, inplace=True)

print(" Missing values handled")

# ==================== HANDLE CATEGORICAL VARIABLES ====================
print("\n Handling categorical variables...")

# Get categorical columns
categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
print(f"Found {len(categorical_cols)} categorical columns")

# Drop categorical columns (avoid unseen label issues)
if len(categorical_cols) > 0:
    print(f"Dropping categorical columns to avoid encoding issues...")
    train = train.drop(categorical_cols, axis=1, errors='ignore')
    test = test.drop(categorical_cols, axis=1, errors='ignore')
    print(f" Dropped {len(categorical_cols)} categorical columns")
else:
    print(" No categorical columns found")

# ==================== FEATURE ENGINEERING ====================
print("\n Feature Engineering...")

# Transaction amount features
if 'TransactionAmt' in train.columns:
    train['Log_Amount'] = np.log1p(train['TransactionAmt'])
    if 'TransactionAmt' in test.columns:
        test['Log_Amount'] = np.log1p(test['TransactionAmt'])

# Time-based features
if 'TransactionDT' in train.columns:
    train['Hour'] = (train['TransactionDT'] / 3600) % 24
    train['DayOfWeek'] = (train['TransactionDT'] / 86400) % 7
    if 'TransactionDT' in test.columns:
        test['Hour'] = (test['TransactionDT'] / 3600) % 24
        test['DayOfWeek'] = (test['TransactionDT'] / 86400) % 7

print(" Features engineered")
print(f"   Train shape: {train.shape}")
print(f"   Test shape: {test.shape}")

# ==================== PREPARE FINAL DATA ====================
print("\n Preparing final datasets...")

# Separate target and features
X_train = train.drop(['isFraud', 'TransactionID'], axis=1, errors='ignore')
y_train = train['isFraud']
X_test = test.drop('TransactionID', axis=1, errors='ignore')

print(f" X_train shape: {X_train.shape}")
print(f" y_train shape: {y_train.shape}")
print(f" X_test shape: {X_test.shape}")

# Check class distribution
print(f"\n Class Distribution:")
print(f"   Legitimate (0): {(y_train == 0).sum():,}")
print(f"   Fraudulent (1): {(y_train == 1).sum():,}")
print(f"   Fraud Rate: {y_train.mean() * 100:.2f}%")

# ==================== SAVE PROCESSED DATA ====================
print("\n Saving processed data...")

X_train.to_csv(r'C:\Users\prachi.patil\Downloads\fraud_detection\data\processed\X_train.csv', index=False)
y_train.to_csv(r'C:\Users\prachi.patil\Downloads\fraud_detection\data\processed\y_train.csv', index=False)
X_test.to_csv(r'C:\Users\prachi.patil\Downloads\fraud_detection\data\processed\X_test.csv', index=False)

print(f" X_train.csv saved ({X_train.shape[0]:,} rows, {X_train.shape[1]} features)")
print(f" y_train.csv saved")
print(f" X_test.csv saved ({X_test.shape[0]:,} rows, {X_test.shape[1]} features)")
print("✅ Saved to data/processed/")

print("\n Preprocessing Complete!")