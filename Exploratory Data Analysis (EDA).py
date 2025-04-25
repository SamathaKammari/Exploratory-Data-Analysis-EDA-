#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO


# In[2]:


df=pd.read_csv("Indian Liver Patient Dataset (ILPD).csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


# Basic Data Cleaning
# Remove duplicates
df = df.drop_duplicates()


# In[7]:


df = df.fillna(df.mean(numeric_only=True))


# In[8]:


# Basic Analysis
print("Dataset Summary:")
print(df.describe())
print("\nPatient Distribution:")
print(df['is_patient'].value_counts(normalize=True))


# In[9]:


# Correlation Analysis
correlation_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Liver Parameters')
plt.tight_layout()
plt.show()


# In[10]:


# Visualization 1: Age Distribution by Patient Status
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='age', hue='is_patient', multiple='stack')
plt.title('Age Distribution by Patient Status')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# In[11]:


# Visualization 2: Box Plot of Total Bilirubin by Gender and Patient Status
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='gender', y='tot_bilirubin', hue='is_patient')
plt.title('Total Bilirubin by Gender and Patient Status')
plt.xlabel('Gender')
plt.ylabel('Total Bilirubin')
plt.show()


# In[20]:


import numpy as np
# Handle missing values (fill numeric with mean, categorical with mode)
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
df['gender'] = df['gender'].fillna(df['gender'].mode()[0])

# Check for invalid values in tot_bilirubin
print("Checking tot_bilirubin for invalid values:")
print("Negative values:", (df['tot_bilirubin'] < 0).sum())


# In[14]:


# Feature Importance Analysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[15]:


# Prepare data for modeling
X = df.drop(['is_patient', 'gender'], axis=1)  # Exclude non-numeric gender for simplicity
y = df['is_patient']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[16]:


# Train Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)


# In[17]:


# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Feature Importance for Predicting Liver Disease')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# Print feature importance
print("\nFeature Importance:")
print(feature_importance)


# In[22]:


# Handle missing values (fill numeric with mean, categorical with mode)
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
df['gender'] = df['gender'].fillna(df['gender'].mode()[0])

# Fix invalid gender entry
df['gender'] = df['gender'].replace('F Pueblo', 'Female')

# Check for negative values in tot_bilirubin
print("Checking for negative values in tot_bilirubin:")
print("Negative values:", (df['tot_bilirubin'] < 0).sum())

# Clip tot_bilirubin to ensure non-negative values
df['tot_bilirubin'] = df['tot_bilirubin'].clip(lower=0)

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# Patterns, Trends, and Anomalies
# 1. Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df[numeric_columns].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Numeric Features')
plt.show()

# 2. Distribution of Key Features by is_patient
key_features = ['tot_bilirubin', 'direct_bilirubin', 'sgpt', 'sgot', 'alkphos']
for feature in key_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='is_patient', y=feature, data=df)
    plt.title(f'Distribution of {feature} by Liver Disease Status')
    plt.show()

# 3. Age Distribution by Gender and is_patient
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='age', hue='is_patient', multiple='stack', bins=20)
plt.title('Age Distribution by Liver Disease Status')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='age', hue='gender', multiple='stack', bins=20)
plt.title('Age Distribution by Gender')
plt.show()

# 4. Scatter Plot with tot_bilirubin as size (fixed for non-negative values)
# Scale tot_bilirubin for better visualization
df['tot_bilirubin_scaled'] = np.clip(df['tot_bilirubin'], 0, None) * 10  # Scale for marker size
fig = px.scatter(df, x='sgpt', y='sgot', size='tot_bilirubin_scaled', 
                 color='is_patient', hover_data=['age', 'gender'], 
                 title='SGPT vs SGOT with Total Bilirubin as Size')
fig.show()

# 5. Outlier Detection
for feature in key_features:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[feature] < Q1 - 1.5 * IQR) | (df[feature] > Q3 + 1.5 * IQR)]


# In[ ]:




