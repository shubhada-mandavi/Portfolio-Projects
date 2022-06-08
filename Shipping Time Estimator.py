#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


shipping = pd.read_csv("shipping_train_2.csv")
shipping.info()


# ### Train-test Splitting

# In[3]:


from sklearn.model_selection import train_test_split
train_set , test_set = train_test_split(shipping, test_size=0.2, random_state=42)
print("Rows in train set :" , len(train_set))
print("Rows in test set :" , len(test_set))


# In[4]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(shipping, shipping['shipment_mode']):
    strat_train_set = shipping.loc[train_index]
    strat_test_set = shipping.loc[test_index]


# ### Looking for correlations

# In[5]:


corr_matrix =shipping.corr()
corr_matrix.head()
corr_matrix['shipping_time'].sort_values(ascending=False)


# In[6]:


strat_train_set.head()


# %matplotlib inline
# from pandas.plotting import scatter_matrix
# shipping.plot(kind="scatter",x='shipment_mode',y='shipping_time', alpha=0.8)
# shipping.plot(kind="scatter",x='shipping_company',y='shipping_time', alpha=0.8)
# shipping.plot(kind="scatter",x='drop_off_point',y='shipping_time', alpha=0.8)clf.score(X_test, y_test)

# In[7]:


shipping = strat_train_set.drop("shipping_time",axis=1)
shipping_labels = strat_train_set["shipping_time"].copy()


# In[8]:


shipping.head()


# ### Creating Pipeline

# In[9]:


from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('std_scaler',StandardScaler())
    ])
shipping_tr = my_pipeline.fit_transform(shipping)


# ### Selecting a desired model

# In[10]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(shipping_tr,shipping_labels)


# In[11]:


shipping_tr = pd.DataFrame(shipping_tr)


# In[12]:


some_data = shipping_tr.iloc[:50]
some_labels = shipping_labels.iloc[:50]


# In[13]:


prepared_data = my_pipeline.transform(some_data)
model.predict(prepared_data)


# In[14]:


list(some_labels)


# In[15]:


from sklearn.metrics import mean_squared_error
shipping_predictions = model.predict(shipping)
rf_mse = mean_squared_error(shipping_labels, shipping_predictions)
rf_rmse = np.sqrt(rf_mse)
rf_mse


# ### Using better evaluation technique - Cross Validation

# In[16]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, shipping, shipping_labels, scoring= "neg_mean_squared_error",cv=10)
rmse_scores = np.sqrt(-scores)
rmse_scores


# In[17]:


def print_scores(scores):
    print("scores:",scores)
    print("mean:", scores.mean())
    print("Standard deviation: ",scores.std() )


# In[18]:


print_scores(rmse_scores)


# ## Saving the model

# In[19]:


from joblib import load,dump
dump(model, 'Assignment.joblib')


# ## Testing the model on Test data

# In[20]:


X_test = strat_test_set.drop('shipping_time',axis=1)
Y_test = strat_test_set["shipping_time"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test,final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse


# ## Final result

# In[21]:


strat_test_set_1 = pd.read_csv("ASSIGNMENT_TESTING.csv")
strat_test_set_1.head()


# In[22]:


X_test_prepared_final = my_pipeline.transform(strat_test_set_1)
final_predictions_1 = model.predict(X_test_prepared_final)
final_rmse


# In[23]:


print(final_predictions_1)


# In[24]:


final_predictions_1 = pd.DataFrame(final_predictions_1)
final_predictions_1.to_csv('result.csv', index=False)


# In[ ]:




