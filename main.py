# TASK 1: Credit Scoring Model:
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# EDA:
df=pd.read_csv('credit_data.csv')
print(df.head())
print(df.describe())
print(df.shape)
print(df.isnull().sum())
print(df.duplicated().sum())
df.info()
df['Debt to Income']=df['Debt']/df['Income']
print(df.head())
# Features:
x=df[['Age','Income','Debt','PaymentHistory','CreditScore','Debt to Income']]
y=df['Creditworthy']
# Train_test_split:
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
# Logistic Regression Model:
model_lg=LogisticRegression()
model_lg.fit(x_train_scaled,y_train)
pred_lg=model_lg.predict(x_test_scaled)
print('Logistic  Regression Predictions:',pred_lg)
# Decision Tree Model:
model_tree=DecisionTreeClassifier(random_state=42)
model_tree.fit(x_train,y_train)
pred_tree=model_tree.predict(x_test)
print('Decision Tree Predictions:',pred_tree)
# Random Forest Model:
model_randomforest=RandomForestClassifier(n_estimators=10,random_state=42)
model_randomforest.fit(x_train,y_train)
pred_randomforest=model_randomforest.predict(x_test)
print('Random Forest Predictions:',pred_randomforest)
# METRICS:
# Accuracy:
accuracy_lg=accuracy_score(y_test,pred_lg)
print('Logistic Regression Accuracy:',accuracy_lg)
accuracy_tree=accuracy_score(y_test,pred_tree)
print('Decision Tree Accuracy:',accuracy_tree)
accuracy_randomforest=accuracy_score(y_test,pred_randomforest)
print('Random Forest Accuracy:',accuracy_randomforest)
# Precision:
precision_lg=precision_score(y_test,pred_lg)
print('Logistic Regression Precision:',precision_lg)
precision_tree=precision_score(y_test,pred_tree)
print('Decision Tree Precision:',precision_tree)
precision_randomforest=precision_score(y_test,pred_randomforest)
print('Random Forest Precision:',precision_randomforest)
# Recall:
recall_lg=recall_score(y_test,pred_lg)
print('Logistic Regression Recall_Score:',recall_lg)
recall_tree=recall_score(y_test,pred_tree)
print('Decision Tree Recall_Score :',recall_tree)
recall_randomforest=recall_score(y_test,pred_randomforest)
print('Random Forest Recall_Score:',recall_randomforest)
# F1_score:
f1_score_lg=f1_score(y_test,pred_lg)
print('Logistic Regression F1_score:',f1_score_lg)
f1_score_tree=f1_score(y_test,pred_tree)
print('Decision Tree F1_score :',f1_score_tree)
f1_score_randomforest=f1_score(y_test,pred_randomforest)
print('Random Forest F1_score:',f1_score_randomforest)
# Roc_Auc_Score:
prob_lg=model_lg.predict_proba(x_test_scaled)[:,1]
roc_auc_score_lg=roc_auc_score(y_test,prob_lg)
print('Logistic Regression Roc_Auc:',roc_auc_score_lg)
prob_tree=model_tree.predict_proba(x_test)[:,1]
roc_auc_score_tree=roc_auc_score(y_test,prob_tree)
print('Decision Tree Roc_Auc:',roc_auc_score_tree)
prob_randomforest=model_randomforest.predict_proba(x_test)[:,1]
roc_auc_score_randomforest=roc_auc_score(y_test,prob_randomforest)
print('Random Forest Roc_Auc:',roc_auc_score_randomforest)







