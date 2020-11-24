#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.linear_model  import Ridge,Lasso,RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_sco


# In[2]:


loan_data = pd.read_csv('C:\\Users\AmEy\personal\Downloads\loan_data.csv')
loan_data.head()


# In[3]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12,16), facecolor='white')
plotnumber = 1

for column in loan_data:
    if plotnumber<=9 :
        ax = plt.subplot(3,3,plotnumber)
        sns.distplot(loan_data[column])
        plt.xlabel(column,fontsize=12)
        #plt.ylabel('Salary',fontsize=20)
    plotnumber+=1
plt.tight_layout()
#plt.show()


# In[4]:


loan_data['good_Loans']=loan_data['bad_loans'].apply(lambda y : 'Yes' if y==0 else 'no')
loan_data.head()


# In[5]:


X = loan_data.drop(['bad_loans','good_Loans'],axis = 1)
y = loan_data['good_Loans']


# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[7]:


print(y_train.shape,y_test.shape)


# In[8]:


model = LogisticRegression()

model.fit(X_train,y_train)


# In[9]:


# r2 score
model.score(X_train,y_train)


# In[10]:


y_pred = model.predict(X_test)


# In[11]:


accuracy=accuracy_score(y_test,y_pred)
accuracy


# In[12]:


conf_mat = confusion_matrix(y_test,y_pred)
conf_mat


# In[13]:


print(classification_report(y_test, y_pred))


# ### DecisionTreeClassifier

# In[14]:


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(X_train,y_train)


# In[15]:


y_pred = model.predict(X_test) # gives output interms of yes/no


# In[16]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print('model accuracy:',accuracy_score(y_test, y_pred)*100)


# 
# ### random forest classifier model

# In[17]:


from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators=150)
RF_model.fit(X_train,y_train)


# In[18]:


y_pred = RF_model.predict(X_test)


# In[19]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))

print('model accuracy:',accuracy_score(y_test,y_pred)*100)

print(classification_report(y_test,y_pred))


# In[20]:


###### on compairing accuracy score of decision tree and random forest ,
# predictions by Random forest are more accurate


