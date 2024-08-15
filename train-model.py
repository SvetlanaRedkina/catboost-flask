#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool


# In[3]:


data_24 = r"C:\Users\sveta\Desktop\catboost\data\data.csv"
data = pd.read_csv(data_24, header=0)
data.head()


# In[4]:


categorical = ['feature_1']


# In[5]:


numeric = ['feature_2', 'feature_3']


# In[6]:


X = data[categorical + numeric]
y = data['label']


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[9]:


train_pool = Pool(data=X_train, label=y_train, cat_features=categorical)
test_pool = Pool(data=X_test, label=y_test, cat_features=categorical)


# In[11]:


model = CatBoostRegressor()


# In[12]:


model.fit(train_pool)


# In[13]:


model.save_model('models/catboost_model.cbm')

