#!/usr/bin/env python
# coding: utf-8
Objective here is to detect whether the customer is active or already left the network.
this can be achieved by using Multiclass logistic regression wherein all the features are used to predict the  chance of attrition
# #### Importing the required Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import seaborn as sns


# In[2]:


import warnings

warnings.filterwarnings('ignore')


# In[3]:


# Let's use the handy function we created
def adj_r2(x,y,r2):
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2


# #### Importing the Dataset

# In[4]:


fiber_bits=pd.read_csv("C:\\Users\Amey\personal\Downloads\Fiberbits.csv")


# ##### Check first five rows of the dataset

# In[5]:


fiber_bits.head()


# #### To see column names

# In[6]:


fiber_bits.columns


# ##### Checking the null values

# In[7]:


fiber_bits.isna().sum()


# ##### checking the shape of the dataset

# In[8]:



print('Number of rows are: ',fiber_bits.shape[0])
print('Number of columns are: ',fiber_bits.shape[1])


# #### To check the correlation between the features and the target column

# In[9]:


sns.heatmap(fiber_bits.corr(),annot=True)


# #### Spliting the dataset into feature and target

# In[10]:


X = fiber_bits.drop(columns = ['active_cust'])
y = fiber_bits['active_cust']


# As the values in the feature columns are of different magnitude they are brought to  uniform scale by using Standardization Technique

# In[11]:


from sklearn.preprocessing import StandardScaler
SC = StandardScaler()
X_scaled = SC.fit_transform(X)


# In[12]:


X_scaled


# To treat Multicolinearity issue ,variance inflaction factor is used ,where features with vif more than 5 are removed 

# In[13]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 
vif = pd.DataFrame()
vif["vif"] = [variance_inflation_factor(X_scaled,i) for i in range(X_scaled.shape[1])]
vif["Features"] = X.columns

#let's check the values
vif


# from the above column it can be seen that none of the feature is exiding the threshold ,thus all features are retained 

# ###### Split feature and target columns into Train & Test data 

# In[14]:


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# In[15]:


x_test.shape,y_test.shape


# ##### Since the target column is categorical ,Classification model can be used ,here we proceed with Logistic Regression 

# In[16]:


model = LogisticRegression()
model.fit(x_train,y_train)


# In[17]:


# r2 score
model.score(x_train,y_train)


# In[18]:


# adj_r2 score

adj_r2(x_train,y_train,model.score(x_train,y_train))


# our adjusted r2 score is almost same as r2 score, thus we are not being penalized for use of many features.
# 
# let's see how well our model performs on the test data set.

# In[19]:


y_pred = model.predict(x_test)
y_pred


# In[20]:


metrics.confusion_matrix(y_test,y_pred)


# In[21]:


metrics.accuracy_score(y_pred,y_test)


# In[22]:


from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score,classification_report

print(classification_report(y_test, y_pred))


# In[23]:


auc = roc_auc_score(y_test, y_pred)
auc


# In[24]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred)


# #### ROC-AUC Curve

# In[25]:


plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc)
#pauc = roc_auc_score(y_test, y_pred)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




