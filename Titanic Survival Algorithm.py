#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[4]:


passengers=pd.read_csv('titanic.csv')


# In[5]:


passengers=pd.read_csv('titanic.csv')


# In[6]:


passengers=pd.read_csv('titanic.csv')


# In[7]:


print(passengers.head())


# In[8]:


passengers['Sex']=passengers['Sex'].map({'male':0,'female':1})


# In[9]:


print(passengers)


# In[10]:


passengers['Age'].fillna(value=round(np.mean(passengers['Age'])),inplace=True)


# In[11]:


passengers['first_class']=passengers['Pclass'].apply(lambda p: 1 if p==1 else 0)


# In[12]:


passengers['second_class']=passengers['Pclass'].apply(lambda p: 1 if p==2 else 0)


# In[13]:


features= passengers[['Sex','Age','first_class','second_class']]


# In[14]:


print(features.head)


# In[15]:


survival=passengers['Survived']


# In[16]:


features_train,features_test,survival_train,survival_test=train_test_split(features, survival,test_size=0.30)


# In[17]:


scaler=StandardScaler()


# In[18]:


features_train=scaler.fit_transform(features_train)


# In[19]:


features_test=scaler.transform(features_test)


# In[20]:


model=LogisticRegression()
model.fit(features_train,survival_train)


# In[21]:


training_score=model.score(features_train,survival_train)
print(training_score)


# In[22]:


test_score=model.score(features_test,survival_test)
print(test_score)


# In[23]:


print(model.coef_)


# In[28]:


Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
Sanjay = np.array([0.0,23.0,1.0,0.0])


# In[29]:


sample_passengers=np.array([Jack,Rose,Sanjay])


# In[30]:


sample_passengers=scaler.transform(sample_passengers)


# In[31]:


print(model.predict(sample_passengers))
print(model.predict_proba(sample_passengers)[:,1])


# In[ ]:





# In[ ]:





# In[ ]:





# # 
