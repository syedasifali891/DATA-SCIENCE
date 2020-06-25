#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
#os.chdir("")


# In[2]:


bank=pd.read_csv("UniversalBank.csv",na_values=["?",","])
print(bank.shape)
print(type(bank))


# In[3]:


print(bank.columns)
print(bank.dtypes)


# In[4]:


bank.head(6)


# In[5]:


bank['Education']=bank['Education'].astype('category')


# In[6]:


bank=pd.get_dummies(bank)


# In[7]:


bank.head(5)


# In[8]:


bank=bank.fillna(bank.mean())


# In[9]:


from sklearn import preprocessing, metrics, model_selection
from sklearn.model_selection import train_test_split


# In[10]:


# Divide in to train and test
y=bank["Personal Loan"]
X=bank.drop('Personal Loan', axis=1)

#from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  


# In[11]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[12]:


from sklearn.preprocessing import StandardScaler


# In[14]:


scaler = StandardScaler()
scaler.fit(X_train)


# In[15]:


X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)


# # Build SVM Classifier

# In[16]:


from sklearn import svm


# In[17]:


model= svm.SVC(C=10,kernel='rbf')
model.fit(X_train,y_train)


# In[18]:



y_pred = model.predict(X_test)


# In[19]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))


# # GridSearch Cross validation

# In[20]:


#from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV
Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
ks = ['linear','poly','rbf','sigmoid']
param_grid = {'C': Cs, 'gamma' : gammas, 'kernel' : ks}
clf = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, n_jobs=4)
clf.fit(X=X_train, y=y_train)
svm_model = clf.best_estimator_
print (clf.best_score_, clf.best_params_)


# In[21]:



y_pred_test=svm_model.predict(X_test)
print(accuracy_score(y_test,y_pred_test))


# Explore function SVM regressor
# http://scikit-learn.org/stable/modules/svm.html

# ## Build Random Forest Classifier

# In[22]:


from sklearn.ensemble import RandomForestClassifier


# In[23]:


clf = RandomForestClassifier(n_estimators=50,max_features=5)
clf.fit(X=X_train, y=y_train)


# In[24]:



y_pred = clf.predict(X_test)


# In[25]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))


# In[ ]:





# In[ ]:




