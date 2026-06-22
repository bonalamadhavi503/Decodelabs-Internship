import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from scipy.stats import zscore

#1dataset creation
np.random.seed(42)
n = 200

df = pd.DataFrame({
    'Age': np.random.randint(18, 25, n),
    'Study_Hours': np.random.randint(1, 10, n),
    'Attendance': np.random.randint(60, 100, n),
    'Marks': np.random.randint(40, 100, n)
})

# Add missing values
df.loc[10, 'Age'] = np.nan
df.loc[20, 'Study_Hours'] = np.nan
df.loc[30, 'Attendance'] = np.nan
df.loc[40, 'Marks'] = np.nan

# Add outliers
df.loc[50, 'Marks'] = 250
df.loc[60, 'Attendance'] = 200
df.loc[70, 'Study_Hours'] = 50

print("\n ORIGINAL DATA ")
print(df.head())

print("\nMissing Values (Before):")
print(df.isnull().sum())

#2
num_cols = df.select_dtypes(include=np.number).columns

# ---- Mean Imputation ----
df_mean = df.copy()
df_mean[num_cols] = df_mean[num_cols].fillna(df_mean[num_cols].mean())
print("\n AFTER MEAN IMPUTATION") 
print(df_mean.head())

# ---- Median Imputation ----
df_median = df.copy()
df_median[num_cols] = df_median[num_cols].fillna(df_median[num_cols].median())
print("\n AFTER MEDIAN IMPUTATION ")
print(df_median.head())

# ---- KNN Imputation (FINAL USED) ----
imputer = KNNImputer(n_neighbors=5)
df[num_cols] = imputer.fit_transform(df[num_cols])

print("\nMissing Values (After KNN):")
print(df.isnull().sum())

#3
Q1 = df[num_cols].quantile(0.25)
Q3 = df[num_cols].quantile(0.75)
IQR = Q3 - Q1

outliers_iqr = ((df[num_cols] < (Q1 - 1.5 * IQR)) |
                (df[num_cols] > (Q3 + 1.5 * IQR)))

print("\nIQR OUTLIERS COUNT")
print(outliers_iqr.sum())

df = df[~outliers_iqr.any(axis=1)]

print("\nShape after IQR removal:", df.shape)

#4
z_scores = np.abs(zscore(df[num_cols]))

outliers_z = (z_scores > 3)

print("\n Z-SCORE OUTLIERS COUNT ")
print(outliers_z.sum())

df = df[(z_scores < 3).all(axis=1)]

print("\nShape after Z-score removal:", df.shape)

# 5
df['Performance_Index'] = (df['Marks'] + df['Attendance']) / 2
df['Study_Efficiency'] = df['Marks'] / (df['Study_Hours'] + 1)
df['Attendance_Study_Score'] = df['Attendance'] * df['Study_Hours']

print("\n FEATURE ENGINEERING DONE ")
print(df[['Performance_Index',
          'Study_Efficiency',
          'Attendance_Study_Score']].head())

#6
print("\n FINAL DATASET ")
print(df.head())

print("\nFinal Shape:", df.shape)

#7
df.to_csv("cleaned_dataset.csv", index=False)

print("\nCleaned dataset saved successfully!")