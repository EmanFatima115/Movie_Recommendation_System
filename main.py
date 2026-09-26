from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from ucimlrepo import fetch_ucirepo
from xgboost import XGBClassifier

heart_disease=fetch_ucirepo(id=45)
x=heart_disease.data.features
y=heart_disease.data.targets

print(x.head())

y=(y['num']>0).astype(int)
print(y.head())
print(x.isnull().sum())
x['ca']=x['ca'].fillna(x['ca'].median())
x['thal']=x['thal'].fillna(x['thal'].median())
print(x.isnull().sum())
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

model_svm=SVC()
model_svm.fit(x_train_scaled,y_train)

model_lg=LogisticRegression()
model_lg.fit(x_train_scaled,y_train)

model_forest=RandomForestClassifier(random_state=42)
model_forest.fit(x_train,y_train)

model_xgb=XGBClassifier(random_state=42)
model_xgb.fit(x_train,y_train)

pred_svm=model_svm.predict(x_test_scaled)
print("SVM Predictions:",pred_svm)
pred_lg=model_lg.predict(x_test_scaled)
print("Logistic Regression Predictions:",pred_lg)
pred_forest=model_forest.predict(x_test)
print("RandomForestClassifier Predictions:",pred_forest)
pred_xgb=model_xgb.predict(x_test)
print("XGB Classifier Predictions:",pred_xgb)
