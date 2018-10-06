
# coding: utf-8

# In[1]:


get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[2]:


#Import the Data to a pandas Datafram
#Import Train Data
traindata = pd.read_csv(r"data\adult.data",
                   names=['Age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','label'])

#Import Test Data
testdata= pd.read_csv(r"data\adult.test",
                   names=['Age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','label'])



# In[3]:


#Append the test and train dataset to one
data = traindata.append(testdata)
#View the data
data.head()


# In[4]:


print("Total Dataset:", data.shape[0])


# In[5]:


# Lets Analyze the Data
data.label.unique() # Looking at the result below there are labels which are inconsistent. - It has to be corrected


# In[6]:


#replace function of the Pandas is used to correct and make the values consistent
data['label']= data['label'].replace(' <=50K.', ' <=50K')
data['label']=data['label'].replace(' >50K.', ' >50K')
data.label.unique() #now the values are consistent


# In[7]:


#from the above results we still see that there is nan value, lets remove it as the dataset without unknow label is not necessary
data = data.dropna()
print("Total Dataset after removing nan:", data.shape[0])


# In[8]:


#Although the Dataset description says the Age is continuous, there are numbers stores as string see below results
data.Age.unique()


# In[9]:


#lets connvert to number and see the result and datatype
data['Age'] = data['Age'].apply(pd.to_numeric)
data.Age.unique()


# In[10]:


#Lets Visualize the Data


# In[11]:


# Use of Pivot function with aggregate function length to get the grouped data with field 'sex' againts for each label
pivot_sex_df = data.pivot_table(index='sex',columns='label',values='Age',aggfunc=len,fill_value=0,margins=True)
pivot_sex_df_percent = pivot_sex_df.div( pivot_sex_df.iloc[:,-1], axis=0 )
pivot_sex_df_percent.iloc[:-1,:-1]


# In[12]:


#Plot the Data
plot1= pivot_sex_df_percent.iloc[:-1,:-1].plot.bar(stacked=True, figsize=(10,7), 
                                            title = "Sex vs Salary Group")
plot1.set(xlabel="Sex", ylabel="Percentage%")


# In[13]:


#Obervation from the above is % of males having Salary above 50K is more compared to Female


# In[14]:


pivot_race_df = data.pivot_table(index='race',columns='label',values='Age',aggfunc=len,fill_value=0,margins=True)


# In[15]:


pivot_race_df_percent = pivot_race_df.div( pivot_race_df.iloc[:,-1], axis=0 )


# In[16]:


pivot_race_df_percent.iloc[:-1,:-1]


# In[17]:


plot2 = pivot_race_df_percent.iloc[:-1,:-1].plot.bar(stacked=True, figsize=(10,7),title = "Race vs Salary Group")
plot2.set(xlabel="Race", ylabel="Percentage%")


# In[18]:


#From the above we can see that People of Asian-Pac_Islander and White get salary more than 50K


# In[19]:


pivot_relationship_df = data.pivot_table(index='relationship',columns='label',values='Age',aggfunc=len,fill_value=0,margins=True)


# In[20]:


pivot_relationship_df_percent = pivot_relationship_df.div( pivot_relationship_df.iloc[:,-1], axis=0 )


# In[21]:


pivot_relationship_df_percent.iloc[:-1,:-1]


# In[22]:


plot3= pivot_relationship_df_percent.iloc[:-1,:-1].plot.bar(stacked=True, figsize=(10,7),title = "Reltionship vs Salary Group")
plot3.set(xlabel="Relationship", ylabel="Percentage%")


# In[23]:


#From the above its clear that relationship of Husband and Wife dominate in having salary more than 50K


# In[24]:


pivot_ms_df = data.pivot_table(index='marital_status',columns='label',values='Age',aggfunc=len,fill_value=0,margins=True)


# In[25]:


pivot_ms_dff_percent = pivot_ms_df.div( pivot_ms_df.iloc[:,-1], axis=0 )


# In[26]:


pivot_ms_dff_percent.iloc[:-1,:-1]


# In[27]:


plot4=pivot_ms_dff_percent.iloc[:-1,:-1].plot.bar(stacked=True, figsize=(10,7),title= "Marital Status vs Salary Group")
plot4.set(xlabel="Marital Status", ylabel="Percentage%")


# In[28]:


#from the above it is very clear that married people are having salary more than 50K


# In[29]:


pivot_occ_df = data.pivot_table(index='occupation',columns='label',values='Age',aggfunc=len,fill_value=0,margins=True)


# In[30]:


pivot_occ_df_percent = pivot_occ_df.div( pivot_occ_df.iloc[:,-1], axis=0 )
pivot_occ_df_percent.iloc[:-1,:-1]


# In[31]:


plot5=pivot_occ_df_percent.iloc[:-1,:-1].plot.bar(stacked=True, figsize=(10,7),title= "Occupation vs Salary Group")
plot5.set(xlabel="Occupation", ylabel="Percentage%")


# In[32]:


#From the above the Executive Managers and Professional Speciality has Salaries more than 50K


# In[33]:


pivot_wc_df = data.pivot_table(index='workclass',columns='label',values='Age',aggfunc=len,fill_value=0,margins=True)


# In[34]:


pivot_wc_df_percent = pivot_wc_df.div( pivot_wc_df.iloc[:,-1], axis=0 )
pivot_wc_df_percent.iloc[:-1,:-1]


# In[35]:


plot6=pivot_wc_df_percent.iloc[:-1,:-1].plot.bar(stacked=True, figsize=(10,7),title= "WorkClass vs Salary Group")
plot6.set(xlabel="WorkClass", ylabel="Percentage%")


# In[36]:


#From the above it is clear that people who are self employed has salary more than 50K


# In[37]:


label = data['label'] # Get the Labels
data = data.drop('label', axis =1) # Drop Label from Data
data['capitals'] = data['capital_gain']+ data['capital_loss'] # convery capital gain and capital loss to one columns
data = data.drop(['capital_gain','capital_loss'], axis =1) 


# In[38]:


data.head()


# In[39]:


#Store the categorical columns in a List
catCol = ['workclass','education','marital_status','occupation','relationship','race','sex','native_country']


# In[40]:


dataLabelEncoder = data.copy(deep=True) #DF for storing Label Encoded Data
le = LabelEncoder()
for item in catCol:#label encoding done for Categorical columns
    le.fit(dataLabelEncoder[item])
    dataLabelEncoder[item]= le.transform(dataLabelEncoder[item])


# In[41]:


dataLabelEncoder.head()


# In[42]:


dataOneHotEncoding = pd.get_dummies(data) #DF for storing One-Hot Encoded data
dataOneHotEncoding.head()


# In[43]:


#Split the Label Encoded Data into training and testing
x_train, x_test, y_train, y_test = train_test_split(dataLabelEncoder,label,test_size = 0.3)


# In[44]:


#Split the Label Encoded Data into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(dataOneHotEncoding,label,test_size = 0.3)


# In[45]:


#Utility Functions


# In[46]:


# Function to make predictions
def prediction(X_test, clf_object):
 
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    #print("Predicted values:")
    #print(y_pred)
    return y_pred


# In[47]:


# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
     
    print("Confusion Matrix: ",
        confusion_matrix(y_test, y_pred))
     
    print ("Accuracy : ",
    accuracy_score(y_test,y_pred)*100)
     
    print("Report : ",
    classification_report(y_test, y_pred))


# In[48]:


# Function to display feature importance in a proper format
def GetTreeFeatureImportance(tree, XDataFrame ):
    feature_importances = pd.DataFrame(tree.feature_importances_,
                                   index = XDataFrame.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
    print("Important Features..")
    print(feature_importances)
    


# # Decision Tree Models - Since the Output is Categorical
# 

# In[49]:


# DECISION TREE (GINI) WITH LABEL ENCODED DATA 


# In[50]:


# Decision tree with gini
clf_gini = DecisionTreeClassifier(criterion = "gini",
            random_state = 100,max_depth=3, min_samples_leaf=5)


# In[51]:


#train the model for LabelEncoded dataset
clf_gini.fit(x_train, y_train)


# In[52]:


clf_gini.score(x_train, y_train) #Training Accuracy


# In[53]:


clf_gini.score(x_test,y_test) # Testing Accuracy


# In[54]:


# Prediction using gini
y_pred_gini = prediction(x_test, clf_gini)
cal_accuracy(y_test, y_pred_gini)


# In[55]:


#Important features Determinig the prediction
GetTreeFeatureImportance(clf_gini, x_train)

For Label Encoded Data
###############################################################################################################################
From the above Decision Tree Prediction model with gini the accuracy we get is 83% and 83% respectively for train and test data
###############################################################################################################################
# In[56]:


# DECISION TREE (GINI) WITH ONE HOT ENCODED DATA 


# In[57]:


#train the model for OneHotEncoded dataset
clf_gini.fit(X_train, Y_train)


# In[58]:


clf_gini.score(X_train, Y_train) #Training Accuracy


# In[59]:


clf_gini.score(X_test,Y_test) # Testing Accuracy


# In[60]:


# Prediction using gini
Y_pred_gini = prediction(X_test, clf_gini)
cal_accuracy(Y_test, Y_pred_gini)


# In[61]:


#Important features Determinig the prediction
GetTreeFeatureImportance(clf_gini, X_train)

For One Hot Encoded Data
###############################################################################################################################
From the above Decision Tree Prediction model with gini the accuracy we get is 84% and 84% respectively for train and test data
###############################################################################################################################
# In[62]:


# DECISION TREE (ENTROPY) WITH LABEL ENCODED DATA 


# In[63]:


# Decision tree with entropy
clf_entropy = DecisionTreeClassifier(
            criterion = "entropy", random_state = 100,
            max_depth = 3, min_samples_leaf = 5)


# In[64]:


#train the model
clf_entropy.fit(x_train, y_train)


# In[65]:


clf_entropy.score(x_train, y_train) #Training accuracy


# In[66]:


clf_entropy.score(x_test, y_test) #Testing accuracy


# In[67]:


# Prediction using entropy
y_pred_entropy = prediction(x_test, clf_entropy)
cal_accuracy(y_test, y_pred_entropy)


# In[68]:


GetTreeFeatureImportance(clf_entropy,x_test)

###############################################################################################################################
From the above Decision Tree Prediction model with entropy the accuracy we get is 83% and 83% respectively for train and test data
###############################################################################################################################
# In[69]:


# DECISION TREE (ENTROPY) WITH ONE HOT ENCODED DATA 


# In[70]:


#train the model
clf_entropy.fit(X_train, Y_train)


# In[71]:


clf_entropy.score(X_train, Y_train) #Training accuracy


# In[72]:


clf_entropy.score(X_test, Y_test) #Testing accuracy


# In[73]:


# Prediction using entropy
Y_pred_entropy = prediction(X_test, clf_entropy)
cal_accuracy(Y_test, Y_pred_entropy)


# In[74]:


GetTreeFeatureImportance(clf_entropy,X_test)

For One Hot Encoded Data
###############################################################################################################################
From the above Decision Tree Prediction model with entropy the accuracy we get is 84% and 84% respectively for train and test data
###############################################################################################################################
# # Random Forest Classifier Models

# In[75]:


# RANDOM FOREST WITH LABEL ENCODED DATA


# In[76]:


rndTree = RandomForestClassifier(n_estimators=10, max_depth = 5)
rndTree.fit(x_train,y_train)


# In[77]:


rndTree.score(x_train,y_train) #Training accuracy


# In[78]:


rndTree.score(x_test,y_test) # Testing accuracy


# In[79]:


# Prediction using Random Forest
Y_pred_rndFor = prediction(x_test, rndTree)
cal_accuracy(y_test, Y_pred_rndFor)


# In[80]:


GetTreeFeatureImportance(rndTree, x_test)

For Label Encoded Data
###############################################################################################################################
From the above Random Forest the accuracy we get is 84% and 84% respectively for train and test data
###############################################################################################################################
# In[81]:


#RANDOM FOREST WITH ONE HOT ENCODED DATA 


# In[82]:


rndTree = RandomForestClassifier(n_estimators=10, max_depth = 5)
rndTree.fit(X_train,Y_train)


# In[83]:


rndTree.score(X_train,Y_train) #Training accuracy


# In[84]:


rndTree.score(X_test,Y_test) # Testing accuracy


# In[85]:


# Prediction using Random Forest
Y_pred_rndFor = prediction(X_test, rndTree)
cal_accuracy(Y_test, Y_pred_rndFor)


# In[86]:


GetTreeFeatureImportance(rndTree, X_test)

For One Hot Encoded Data
###############################################################################################################################
From the above Random Forest the accuracy we get is 82% and 83% respectively for train and test data
###############################################################################################################################
# # LOGISTIC REGRESSION Classification Model

# In[87]:


#LOGISTIC REGRESSION WITH LABEL ENCODED DATA


# In[88]:


logReg = LogisticRegression(max_iter=500)


# In[89]:


logReg.fit(x_train, y_train)


# In[90]:


logReg.score(x_train, y_train) #Training accuracy


# In[91]:


logReg.score(x_test, y_test) #Testing accuracy


# In[92]:


# Prediction using Logistic Regression
Y_pred_logreg = prediction(x_test, logReg)
cal_accuracy(y_test, Y_pred_logreg)


# In[93]:


impFeatures=pd.DataFrame({'Features' :x_test.columns,'Importance': logReg.coef_[0][:]})
impFeatures.sort_values('Importance', ascending =False)

For Label Encoded Data
###############################################################################################################################
From the above Logistic Regression the accuracy we get is 78% and 78% respectively for train and test data
###############################################################################################################################################################################################################################################################################################################################################################################################
# In[94]:


####################### LOGISTIC REGRESSION WITH ONE HOT ENCODED DATA ###################################


# In[95]:


logReg = LogisticRegression(max_iter=500)


# In[96]:


logReg.fit(X_train, Y_train)


# In[97]:


logReg.score(X_train, Y_train) #Training Score


# In[98]:


logReg.score(X_test, Y_test) # Testing Score


# In[99]:


# Prediction using Logistic Regression
Y_pred_logreg = prediction(X_test, logReg)
cal_accuracy(Y_test, Y_pred_logreg)


# In[100]:


impFeatures=pd.DataFrame({'Features' :X_test.columns,'Importance': logReg.coef_[0][:]})
impFeatures.sort_values('Importance', ascending =False)

For One Hot Encoded Data
###############################################################################################################################
From the above Logistic Regression the accuracy we get is 79% and 79% respectively for train and test data
###############################################################################################################################################################################################################################################################################################################################################################################################<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
All model Scores
Logistic Regression - OneHot - Train: 79% - Test: 79%
Logistic Regression - Label - Train: 78% - Test: 78%
Random Forest - OneHot - Train: 82% - Test: 83%
Random Forest - Label - Train: 84% - Test: 84%
Decision Tree(entropy) - OneHot - Train: 84% - Test: 84%
Decision Tree(entropy) - Label - Train: 83% - Test: 83%
Decision Tree(gini) - OneHot - Train: 84% - Test: 84%
Decision Tree(gini) - Label - Train: 83% - Test: 83%
# # Problem Statement
# 

# # Problem 1:
# # Prediction task is to determine whether a person makes over 50K a year.
VARIOUS MODELS USED TO PREDICT THE SALARY GROUP
LOGISTIC REGRESSION 
DECISION TREE
RANDOM FOREST
ALL ABOVE MODEL IS USED WITH LABEL ENCODER AND ONE HOT ENCODED DATA
# # Problem 2:
# # Which factors are important
Below feature are identifies to be major contributor to  determine if the person is making >50K or <=50K

Relationship
education num
Age
capitals
workclass
hours_per_week
# # Problem 3:
# # Which algorithms are best for this dataset
From the Prediction accuraies its clear that Decision Tree and Random Forest are giving better results.
Since Random Forest are collection of Decision Trees. Decision Tree can be determined as the best Model