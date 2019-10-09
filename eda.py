import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler

import matplotlib
font = {'family': 'normal', 'size': 14}
matplotlib.rc('font', **font)

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', 500)

here_path = os.path.dirname(os.path.realpath(__file__))

train_features_file_name = 'data.csv'
train_features_file_path = os.path.join(here_path, train_features_file_name)

train_df = pd.read_csv(train_features_file_path)

print(train_df.shape)

# Data info ............................................................................................................
print(train_df.shape)

print(train_df.head())

train_df.loc[train_df['churn'] == 'Yes', 'churn'] = 1
train_df.loc[train_df['churn'] == 'No', 'churn'] = 0
# ......................................................................................................................


# Missing data .........................................................................................................
missing_stat_df = pd.DataFrame({'Dtype': train_df.dtypes, 'Unique values': train_df.nunique(),
                                'Number of Missing values': train_df.isnull().sum(),
              'Percentage Missing': (train_df.isnull().sum() / len(train_df)) * 100
                                }).sort_values(by='Number of Missing values', ascending=False)

print(missing_stat_df)
# ......................................................................................................................

cols_numeric = ['tenure', 'MonthlyCharges', 'TotalCharges']

cols_cat = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod']

# ----------------------------------------------------------------------------------------------------------------------
# Gender
unique_values = train_df['gender'].unique()
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
labels = ['Churned', 'Stayed']
sizes = [len(train_df[train_df['churn']==1]), len(train_df[train_df['churn']==0])]
colors = ['#ff6666', '#99ff99']

plt.pie(sizes, labels=labels, colors=colors, startangle=90, autopct='%1.0f%%', labeldistance=1.05)
centre_circle = plt.Circle((0, 0), 0.4, color='black', fc='white', linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.axis('equal')
plt.tight_layout()

plt.subplot(1,3,2)
plt.title('Churned')
sizes = [len(train_df[(train_df['churn']==1) & (train_df['gender']==unique_values[0])]),
                len(train_df[(train_df['churn']==1) & (train_df['gender']==unique_values[1])])]
colors = ['#ffb3e6', '#c2c2f0']

plt.pie(sizes, labels=unique_values, colors=colors, startangle=90, autopct='%1.0f%%', labeldistance=0.8)
centre_circle = plt.Circle((0, 0), 0.4, color='black', fc='white', linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.axis('equal')
plt.tight_layout()

plt.subplot(1,3,3)
plt.title('Stayed')
sizes = [len(train_df[(train_df['churn']==0) & (train_df['gender']==unique_values[0])]),
                len(train_df[(train_df['churn']==0) & (train_df['gender']==unique_values[1])])]

plt.pie(sizes, labels=unique_values, colors=colors, startangle=90, autopct='%1.0f%%', labeldistance=0.8)
centre_circle = plt.Circle((0, 0), 0.4, color='black', fc='white', linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.axis('equal')
plt.tight_layout()
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Contract
unique_values = train_df['Contract'].unique()
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
labels = ['Churned', 'Stayed']
sizes = [len(train_df[train_df['churn']==1]), len(train_df[train_df['churn']==0])]
colors = ['#ff6666', '#99ff99']

plt.pie(sizes, labels=labels, colors=colors, startangle=90, autopct='%1.0f%%', labeldistance=1.05)
centre_circle = plt.Circle((0, 0), 0.4, color='black', fc='white', linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.axis('equal')
plt.tight_layout()

plt.subplot(1,3,2)
plt.title('Churned')
sizes = [len(train_df[(train_df['churn']==1) & (train_df['Contract']==unique_values[0])]),
         len(train_df[(train_df['churn']==1) & (train_df['Contract']==unique_values[1])]),
         len(train_df[(train_df['churn']==1) & (train_df['Contract']==unique_values[2])])]
colors = ['#ffb3e6', '#c2c2f0', '#ffcc99']

plt.pie(sizes, labels=unique_values, colors=colors, startangle=90, autopct='%1.0f%%', labeldistance=1.0)
centre_circle = plt.Circle((0, 0), 0.4, color='black', fc='white', linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.axis('equal')
plt.tight_layout()

plt.subplot(1,3,3)
plt.title('Stayed')
sizes = [len(train_df[(train_df['churn']==0) & (train_df['Contract']==unique_values[0])]),
         len(train_df[(train_df['churn']==0) & (train_df['Contract']==unique_values[1])]),
         len(train_df[(train_df['churn']==0) & (train_df['Contract']==unique_values[2])])]

plt.pie(sizes, labels=unique_values, colors=colors, startangle=90, autopct='%1.0f%%', labeldistance=0.8)
centre_circle = plt.Circle((0, 0), 0.4, color='black', fc='white', linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.axis('equal')
plt.tight_layout()
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Tenure
plt.figure(figsize=(15,5))
sns.distplot(train_df[train_df['churn']==0]['tenure'], bins=100, hist=True, kde=True, rug=False, label='Stayed', color='#99ff99')
sns.distplot(train_df[train_df['churn']==1]['tenure'], bins=100, hist=True, kde=True, rug=False, label='Churned', color='#ff6666')
plt.xlim([0,80])
plt.xlabel('Tenure')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()

# Total Charges
plt.figure(figsize=(15,5))
sns.distplot(train_df[train_df['churn']==0]['TotalCharges'], bins=100, hist=True, kde=True, rug=False, label='Stayed', color='#99ff99')
sns.distplot(train_df[train_df['churn']==1]['TotalCharges'], bins=100, hist=True, kde=True, rug=False, label='Churned', color='#ff6666')
plt.xlim([0,9000])
plt.xlabel('Total Charges')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
# ----------------------------------------------------------------------------------------------------------------------


data = train_df.copy()
target = data['churn']
ids = data['customerID']
df = data.copy()

#  One hot encoding ....................................................................................................
train_df_features_sub = train_df[cols_numeric + cols_cat]

df_features_sub = df[cols_numeric + cols_cat]


def conv_one_hot(data):
    categorial_variables = data.select_dtypes(exclude=['int64', 'float64', 'bool']).columns
    print(type(categorial_variables))
    print("The list of Categorical variables")
    print(list(categorial_variables))
    # Convert Categorical variables to dummies
    cat_var = pd.get_dummies(data[list(categorial_variables)],drop_first=True)
    # Remove originals
    data = data.drop(categorial_variables,axis=1)
    data = pd.concat([data,cat_var],axis=1)
    # removing duplicate columns - useful in case two variables are closely related.
    _, i = np.unique(data.columns, return_index=True)
    data = data.iloc[:, i]
    return data

train_df_features_sub = conv_one_hot(train_df_features_sub)
df_features_sub = conv_one_hot(df_features_sub)
# ......................................................................................................................

# Feature Scaling ......................................................................................................
sc = StandardScaler()

train_matrix = sc.fit_transform(train_df_features_sub[cols_numeric])

train_df_features_sub_standardized = train_df_features_sub.copy()
df_features_sub_standardized = df_features_sub.copy()

train_df_features_sub_standardized[cols_numeric] = sc.transform(train_df_features_sub_standardized[cols_numeric])
df_features_sub_standardized[cols_numeric] = sc.transform(df_features_sub_standardized[cols_numeric])

# train_df_features_sub_standardized.loc[:,cols_numeric] = train_df_features_sub_standardized.loc[:,cols_numeric].transform(lambda x: (x - x.mean()) / x.std())

train_df_features_sub_standardized['churn'] = train_df['churn']
train_df_features_sub_standardized['customerID'] = train_df['customerID']

df_features_sub_standardized['churn'] = target
df_features_sub_standardized['customerID'] = ids

# Extract train and test for final write
train_df_features_sub_standardized = df_features_sub_standardized.iloc[:train_df.shape[0],:]
# test_df_features_sub_standardized = df_features_sub_standardized.iloc[train_df.shape[0]:,:]

print(train_df_features_sub_standardized.shape)
# print(test_df_features_sub_standardized.shape)

train_df_features_sub_standardized_file_name = 'train_df_features_sub_standardized.csv'
train_df_features_sub_standardized_file_path = os.path.join(here_path, train_df_features_sub_standardized_file_name)
train_df_features_sub_standardized.to_csv(train_df_features_sub_standardized_file_path, index=False)

print('end')

plt.show()
