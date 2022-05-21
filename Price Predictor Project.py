#!/usr/bin/env python
# coding: utf-8

# # # Real Estate Price Predictor ##

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


housing = pd.read_csv("data.csv")
housing.head()
housing.info()
housing.describe()


# In[3]:


housing['CHAS'].value_counts()


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
#housing.hist(bins=50, figsize=(20,15))


# # Train-test Splitting 

# def train_test_split(data, test_ratio):
#     np.random.seed(42)
#     shuffled = np.random.permutation(len(data))
#     test_set_size = int(len(data) *test_ratio)
#     test_indices = shuffled[:test_set_size]
#     train_indices = shuffled[test_set_size:]
#     return data.iloc[train_indices],data.iloc[test_indices] 
# 
# train_set , test_set = train_test_split(housing, 0.2)
# print(len(train_set))
# print(len(test_set))

# In[5]:


from sklearn.model_selection import train_test_split
train_set , test_set = train_test_split(housing, test_size=0.2, random_state=42)
print("Rows in train set :" , len(train_set))
print("Rows in test set :" , len(test_set))


# In[6]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[7]:


housing =  strat_train_set.copy()


# ## LOOKING FOR CORRELATIONS

# In[8]:


corr_matrix = housing.corr()
corr_matrix.head()
corr_matrix['MEDV'].sort_values(ascending=False)


# from pandas.plotting import scatter_matrix
# attributes =['RM','ZN','MEDV','LTSTAT']
# scatter_matrix(housing[attributes], figsize=(12,8))

# In[9]:


housing.plot(kind="scatter",x='RM',y='MEDV', alpha=0.8)


# In[10]:


housing.describe()


# ## Trying out Attribute Combinations

# In[11]:


housing["TAXRM"] = housing["TAX"]/housing['RM']
housing["TAXRM"]


# In[12]:


housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()


# ## To Deal with Missing Attributes
# 

#  There are 5 ways to deal with missing attributes:
#  1. Get rid of the missing data points
#  2. Get rid of the attributes
#  3. Set the value to some value like zero, mean or median
#  
#  : housing.dropna(subset=["RM"])]

# In[13]:


#Option 1 #original dataframe remains unchanged  
a=housing.dropna(subset=["RM"])
a.shape


# In[14]:


#Option 2 #original dataframe remains unchanged 
housing.drop('RM', axis=1)


# In[15]:


#option 3 #original dataframe remains unchanged 
median = housing['RM'].median()
housing["RM"].fillna(median)


# In[ ]:





# In[16]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
imputer.fit(housing)


# In[17]:


imputer.statistics_


# In[18]:


X= imputer.transform(housing)
housing_tr = pd.DataFrame(X, columns = housing.columns)


# In[19]:


housing_tr.describe()


# ## Creating Pipeline

# In[20]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy= 'median')),
    ('std_scaler',StandardScaler())
    
])


# In[21]:


housing_num_tr = my_pipeline.fit_transform(housing)


# In[22]:


housing_num_tr.shape


# ## Selecting a desired model 

# ### for Linear Regression model
# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# model.fit(housing_num_tr,housing_labels)

# ###for decision tree model
# from sklearn.tree import DecisionTreeRegressor
# model = DecisionTreeRegressor()
# model.fit(housing_num_tr,housing_labels)

# In[23]:


### For RandomForest Model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(housing_num_tr,housing_labels)


# In[24]:


some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]


# In[25]:


prepared_data = my_pipeline.transform(some_data)
model.predict(prepared_data)


# In[26]:


list(some_labels)


# ## Evaluating the model

# In[27]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)


# ### for Linear Regression model
# 
# lin_mse = mean_squared_error(housing_labels, housing_predictions)
# lin_rmse = np.sqrt(lin_mse)
# lin_mse

# #for decision tree model
# mse = mean_squared_error(housing_labels, housing_predictions)
# rmse = np.sqrt(mse)
# mse

# In[28]:


#for RandomForest model
rf_mse = mean_squared_error(housing_labels, housing_predictions)
rf_rmse = np.sqrt(rf_mse)
rf_mse


# USING BETTER EVALUATION TECHNIQUE - CROSS VALIDATION

# In[29]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring= "neg_mean_squared_error",cv=10)
rmse_scores = np.sqrt(-scores)
rmse_scores


# In[30]:


def print_scores(scores):
    print("scores:",scores)
    print("mean:", scores.mean())
    print("Standard deviation: ",scores.std() )


# In[31]:


print_scores(rmse_scores)


# ## Saving the model

# In[35]:


from joblib import load,dump
dump(model, 'Real_estate.joblib')


# ## Testing the model on Test data

# In[36]:


X_test = strat_test_set.drop('MEDV',axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test,final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse


# In[37]:


print(final_predictions,list(Y_test))


# In[38]:


prepared_data[0]


# ## Using the model

# In[41]:


from joblib import load, dump
import numpy as np
model = load('Real_estate.joblib')
features =np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
       -0.24531453, -1.31238772,  2.61111401, -1.0016859 , -0.5778192 ,
       -0.97491834,  0.41164221, -0.86091034]])
model.predict(features)


# In[ ]:




