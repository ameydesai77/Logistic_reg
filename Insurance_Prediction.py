#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression - Use Product data set

# Logistic Regression - Use Product data set
# 1. Build a predictive model for Bought vs Age
# 2. If Age is 4 then will that customer buy the product?
# 3. If Age is 105 then will that customer buy the product?

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


Prod=pd.read_csv('C:\\Users\Amey\personal\Downloads\Product_sales.csv')
Prod.head()


# In[3]:


Prod.isna().sum()


# In[4]:


X = Prod[['Age']]
X.shape


# In[5]:


y=Prod['Bought']
y.shape


# In[6]:


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# In[7]:


x_test.shape,y_test.shape


# ### Building a predictive model for Bought vs Age

# In[8]:


Log_reg=LogisticRegression()


# In[9]:


Log_reg.fit(x_train,y_train)


# In[10]:


pred_y=Log_reg.predict(x_test)


# In[11]:


metrics.accuracy_score(pred_y,y_test)


# In[12]:


score = Log_reg.score(x_test,y_test)
print(score)


# In[13]:


metrics.f1_score(pred_y,y_test)


# In[14]:


y_test[1],pred_y[1] # Actual data and predicted value for age=1


# In[15]:


pred_y[1]


# ### 2. If Age is 4 then will that customer buy the product?

# In[16]:


pred_y[4] 


# ### If Age is 105 then will that customer buy the product?

# In[17]:


pred_y[105] 


# In[18]:


cm = metrics.confusion_matrix(y_test,pred_y)
cm


# In[19]:


metrics.precision_score(y_test,pred_y)


# In[20]:


import seaborn as sns
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt =".2g", linewidths=0, square = True, cmap = 'Blues_r')
plt.ylabel("Actual label")
plt.xlabel("Predictied lable")
#all_sample_title = "Accuracy score:" ,{0}.format(score)
#plt.title(all_sample_title, size = 15);


# In[21]:


plt.title('Age and Policy')
plt.xlabel('Age')
plt.ylabel('Policy')
plt.scatter(x_train,y_train,color='red')
plt.plot(x_test,Log_reg.predict(x_test),color='blue')
plt.show()


# In[22]:


score = Log_reg.score(x_test,y_test)
print(score)


# In[23]:


#Checking with confusion matrix

cm = metrics.confusion_matrix(y_test,pred_y)
print(cm)


# In[24]:


#Visualization of the confusion matrix

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt =".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel("Actual label");
plt.xlabel("Predictied lable");


# In[ ]:





# In[ ]:




