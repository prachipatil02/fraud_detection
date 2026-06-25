# ==================== IMPORTS ====================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("LOADING IEEE-CIS FRAUD DETECTION DATASET")
print("=" * 70)

# ==================== LOAD DATA ====================
print("\n📂 Loading datasets...")

train_transaction = pd.read_csv(r'C:\Users\prachi.patil\Downloads\fraud_detection\data\raw\train_transaction.csv')
train_identity = pd.read_csv(r'C:\Users\prachi.patil\Downloads\fraud_detection\data\raw\train_identity.csv')
test_transaction = pd.read_csv(r'C:\Users\prachi.patil\Downloads\fraud_detection\data\raw\test_transaction.csv')
test_identity = pd.read_csv(r'C:\Users\prachi.patil\Downloads\fraud_detection\data\raw\test_identity.csv')

print(f"train_transaction: {train_transaction.shape}")
print(f"train_identity: {train_identity.shape}")
print(f"test_transaction: {test_transaction.shape}")
print(f"test_identity: {test_identity.shape}")

# ==================== MERGE DATA ====================
print("\n Merging transaction and identity data...")

train = train_transaction.merge(train_identity, on='TransactionID', how='left')
test = test_transaction.merge(test_identity, on='TransactionID', how='left')

print(f"Combined training set: {train.shape}")
print(f"Combined test set: {test.shape}")

# ==================== EXPLORATORY DATA ANALYSIS ====================
print("\n" + "=" * 70)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# Display basic info
print("\nTraining Set Info:")
print(train.info())

print("\nStatistical Summary:")
print(train.describe())

# ==================== TARGET DISTRIBUTION ====================
print("\nTarget Distribution:")
print(train['isFraud'].value_counts())
print(f"\nFraud Rate: {train['isFraud'].mean() * 100:.2f}%")

# ==================== MISSING VALUES ====================
print("\nMissing Values:")
missing = train.isnull().sum()
missing_pct = (missing / len(train) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Percentage': missing_pct
})
print(missing_df[missing_df['Missing Count'] > 0].sort_values('Percentage', ascending=False))

# ==================== DATA TYPES ====================
print("\nData Types:")
print(train.dtypes.value_counts())

# ==================== SAVE MERGED DATA ====================
print("\n  Saving merged datasets...")
train_output = os.path.join(r'C:\Users\prachi.patil\Downloads\fraud_detection\data\processed', 'train_merged.csv')
test_output = os.path.join(r'C:\Users\prachi.patil\Downloads\fraud_detection\data\processed', 'test_merged.csv')

print(f"\n   Saving {train_output}...")
train.to_csv(train_output, index=False)

print(f"\n   Saving {test_output}...")
test.to_csv(test_output, index=False)

print("✅ Saved to data/processed/")

print("\n Data Exploration Complete!")



