#!/usr/bin/env python
# coding: utf-8

# In[1]:


### IMPORTING LIBRARIES


# In[2]:


import pandas as pd                       ## For data manipulation, cleaning, and analysis using DataFrames.
import numpy as np                        ## For efficient numerical operations and mathematical functions.
import seaborn as sns                     ## For creating static and statistical data visualizations. 
import matplotlib.pyplot as plt
from sklearn.svm import SVC


# In[3]:


import os
working_directory = os.getcwd()
print(working_directory)


# In[4]:


path = working_directory + '/Desktop/CAPSTONE MAIN - insurance/data.csv'
insurance = pd.read_csv(path)


# In[5]:


### DATA COLLECTION AND INSPECTION


# In[6]:


insurance.head(10)


# In[7]:


insurance.info()


# In[8]:


insurance.describe()


# In[9]:


insurance.describe().T


# In[10]:


insurance.shape


# In[11]:


### DATA CATEGORISING 


# In[12]:


## Selecting numerical features

numerical_data = insurance.select_dtypes(include='number')
numerical_features=numerical_data.columns.tolist()
print(f'There are {len(numerical_features)} numerical features:', '\n')
print(numerical_features)


# In[13]:


## Selecting categorical features

categorical_data=insurance.select_dtypes(include= 'object')
categorical_features=categorical_data.columns.tolist()
print(f'There are {len(categorical_features)} categorical features:', '\n')
print(categorical_features)


# In[14]:


categorical_data.describe(include='object').T


# In[15]:


## Variance


# In[16]:


numerical_data.var()


# In[17]:


## Skewness


# In[18]:


numerical_data.skew()


# In[19]:


## Checking for outliers using histogram


# In[20]:


numerical_data.hist(figsize=(25,20),bins=20, color='navy', edgecolor='cyan')
plt.show()


# In[21]:


### DATA CLEANING


# In[22]:


insurance.columns


# In[23]:


insurance.isnull().sum()


# In[24]:


insurance.duplicated().sum()


# In[25]:


Insurance = insurance.drop(['id','Region_Code'], axis=1, inplace=False) 


# In[26]:


Insurance.head(10)


# In[27]:


Insurance.nunique()


# In[28]:


### EDA (EXPLORATORY DATA ANALYSIS)


# In[29]:


## with "Insurance" - new with dropped columns


# In[30]:


Insurance['Vehicle_Age'].value_counts()


# In[31]:


plt.figure(figsize=(8, 6))
sns.countplot(x='Vehicle_Age', data=Insurance, hue='Vehicle_Age', palette='hot', legend=False)
plt.title('Count Plot of Vehicle Age')
plt.show()


# In[32]:


sns.color_palette('magma')


# In[33]:


## with "insurance" - original dataset


# In[34]:


## Bar plots of unqiue value counts in each variable

for col in categorical_features:
    print()
    print(f"\033[1m{col}\033[0m\n") # print column name above the plot
    colors = ['seagreen', 'gold', 'darkred']
    categorical_data[col].value_counts().sort_index().plot(kind='bar', rot=0, xlabel=col, ylabel='count', color=colors)
    plt.show()


# In[35]:


## CORRELATION MATRIX


# In[88]:


from pandas import set_option
set_option("display.precision",3)
correlation=insurance.corr(method='pearson')
correlation


# In[89]:


fig, ax = plt.subplots(figsize=(15, 9))
sns.heatmap(insurance.corr(), ax=ax, annot=True)
plt.show()


# In[37]:


numerical_data.plot(kind='density',figsize=(14,14),subplots=True,layout=(6,2),title="Density plot of Numerical features",sharex=False)
plt.show()


# In[38]:


insurance.columns


# In[39]:


# Relationship between Policy Sales Channel and Response


# In[40]:


sns.boxplot(x='Response', y='Policy_Sales_Channel', data=insurance, palette='gist_heat', hue = 'Response')


# In[41]:


plt.figure(figsize=(10, 8))
sns.violinplot(x="Previously_Insured", y="Vintage", data=insurance, palette='nipy_spectral', alpha=0.7, hue ='Response')

plt.title('Violin Plot')
plt.xlabel('Response')
plt.ylabel('Policy_Sales_Channel')
plt.show()


# In[42]:


## Label Encdoing


# In[43]:


categorical_features


# In[44]:


# Encoding 'Gender' to numerical values
insurance['Gender'] = insurance['Gender'].map({'No': 0, 'Yes': 1})

# Encoding 'Vehicle_Age' column to numeric values
insurance['Vehicle_Age'] = insurance['Vehicle_Age'].map({'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2})

# Encoding 'Vehicle_Damage' column to numeric values
insurance['Vehicle_Damage'] = insurance['Vehicle_Damage'].map({'No': 0, 'Yes': 1})


# In[45]:


## TRAIN TEST SPLIT


# In[46]:


from sklearn.model_selection import train_test_split


# In[47]:


Y = insurance['Response']


# In[48]:


X = insurance[['Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Policy_Sales_Channel']]


# In[49]:


Y.shape


# In[50]:


X.shape


# In[51]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)


# In[52]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[53]:


## Scaling


# In[54]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[55]:


## Hyper parameter tuning


# In[56]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=4, n_informative=4, n_classes=2, n_redundant=0, random_state=42)

c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}

logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)
logreg_cv.fit(X, y)

print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print("Best score is {}".format(logreg_cv.best_score_))


# In[57]:


## Evaluating unseen data


# In[58]:


best_model = logreg_cv.best_estimator_
test_accuracy = best_model.score(X_test.values, y_test)
print(f"Test Set Accuracy: {test_accuracy}")


# In[ ]:





# In[59]:


## Model deployment and fitting


# In[60]:


from sklearn.ensemble import RandomForestClassifier


# In[61]:


model = RandomForestClassifier(n_estimators=100, random_state=42)


# In[62]:


model.fit(X_train, y_train)


# In[63]:


y_pred = model.predict(X_test)


# In[64]:


from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score


# In[65]:


accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average=None)


# In[66]:


print(f"Accuracy: {accuracy}")
print(f"Conf_matrix: {conf_matrix}")
print(f"Class_report: {class_report}")
print(f"F1_Score: {f1}")


# In[67]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)

print(f"Scores for each fold: {scores}")
print(f"Average Accuracy: {scores.mean():.4f}")


# In[68]:


from sklearn.linear_model import LogisticRegression


# In[69]:


model_2 = LogisticRegression()


# In[70]:


model_2.fit(X_train, y_train)


# In[71]:


y_pred_lr = model_2.predict(X_test)


# In[72]:


accuracy = accuracy_score(y_test, y_pred_lr)
conf_matrix = confusion_matrix(y_test, y_pred_lr)
class_report = classification_report(y_test, y_pred_lr, zero_division=0.0)
f1 = f1_score(y_test, y_pred_lr, average=None)


# In[73]:


print(f"Accuracy: {accuracy}")
print(f"Conf_matrix: {conf_matrix}")
print(f"Class_report: {class_report}")
print(f"F1_Score: {f1}")


# In[74]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_digits


# In[75]:


gbc = GradientBoostingClassifier(n_estimators=300,
                                 learning_rate=0.05,
                                 random_state=100,
                                 max_features=5 )


# In[76]:


gbc.fit(X_train, y_train)


# In[77]:


y_pred_gbc = gbc.predict(X_test)


# In[78]:


acc = accuracy_score(y_test, y_pred_gbc)
conf_matrix2 = confusion_matrix(y_test, y_pred_gbc)
class_report2 = classification_report(y_test, y_pred_gbc, zero_division=0.0)
f1_2 = f1_score(y_test, y_pred_gbc, average=None)


# In[79]:


print(f"Accuracy: {acc}")
print(f"Conf_matrix: {conf_matrix2}")
print(f"Class_report: {class_report2}")
print(f"F1_Score: {f1_2}")


# In[80]:


print(f"Gradient Boosting Classifier accuracy is : {acc}")


# In[81]:


from sklearn.tree import DecisionTreeClassifier, plot_tree


# In[82]:


DTclf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
DTclf.fit(X_train, y_train)


# In[83]:


y_pred_dtc = DTclf.predict(X_test)


# In[84]:


print(f"Accuracy: {accuracy_score(y_test, y_pred_dtc):}")
print(f"Conf_matrix: {confusion_matrix(y_test, y_pred_dtc)}")
print(f"Class_report: {classification_report(y_test, y_pred_dtc, zero_division=0.0)}")
print(f"F1_Score: {f1_score(y_test, y_pred_dtc, average=None)}")


# In[90]:


# accuracy score
rfc_acc = accuracy_score(y_test, y_pred)
lr_acc  = accuracy_score(y_test, y_pred_lr)
gbc_acc = accuracy_score(y_test, y_pred_gbc)
dtc_acc = accuracy_score(y_test, y_pred_dtc)

print('RFC: ', rfc_acc)
print('LR: ', lr_acc)
print('GBC: ', gbc_acc)
print('DTC: ', dtc_acc)


# In[ ]:





# In[86]:


### FINAL MODEL selection and Saving of the model


# In[96]:


from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")


# In[97]:


param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'lbfgs']
    # 'class_weigh': ['balanced', None] --imbalance dataset
}

grid_search = GridSearchCV(model_2, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)
grid_search.best_params_

best_params = grid_search.best_params_

lr_final = LogisticRegression(**best_params)
lr_final.fit(X, y)


# In[87]:


import streamlit as st
import pickle


# In[98]:


dump_file = 'lr_final.pkl'
with open(dump_file, 'wb') as f:
    pickle.dump(lr_final, f)


# In[99]:


import os
print(os.getcwd())


# In[ ]:





# In[ ]:


### IQR


# In[ ]:




