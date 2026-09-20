from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

# Load Iris Dataset:
iris=load_iris()
#Create DataFrame:
df=pd.DataFrame(iris.data,columns=iris.feature_names)

df['species']=iris.target
# EDA
print('First 5 rows are:')
print(df.head())

print(df.tail())
print('Shape:')
print(df.shape)
print('Information:')
print(df.info())
print('Duplicate rows are:')
print(df.duplicated().sum())
print('Missing values:')
print(df.isnull().sum())
print('Statistical Summary')
print(df.describe())
print('Species distribution :')
print(df['species'].value_counts())
# Add Species Names:
df['species_name']=df['species'].map({
    0:'setosa',
    1:'versicolor',
    2:'virginica'
})
print(df.head())
sns.pairplot(df,vars=iris.feature_names,
hue='species_name')
plt.show()
x=df[iris.feature_names]
y=df['species']
#Train_Test Split:
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#Logistic Regression:
model_lr=LogisticRegression()
model_lr.fit(x_train,y_train)
pred_lr=model_lr.predict(x_test)
print('Logistic Regression Predictions:')
print(pred_lr)
# Accuracy(lr):
accuracy_lr=accuracy_score(y_test,pred_lr)
print('Logistic Regression Accuracy:')
print(accuracy_lr)
# Decision Tree:
model_dt=DecisionTreeClassifier(random_state=42)
model_dt.fit(x_train,y_train)
pred_dt=model_dt.predict(x_test)
print("Decision Tree Predictions: ")
print(pred_dt)

accuracy_dt=accuracy_score(y_test,pred_dt)
print('Decision Tree Accuracy:')
print(accuracy_dt)
# Accuracy Comparison:
print('Logistic Regression Accuracy:',accuracy_lr)
print('Decision Tree Accuracy:',accuracy_dt)
# Confusion Matrix:
cm_lr=confusion_matrix(y_test,pred_lr)
disp_lr=ConfusionMatrixDisplay(confusion_matrix=cm_lr,display_labels=iris.target_names)
disp_lr.plot()
plt.title('Logistic Regression Confusion Matrix')
plt.show()

# Misclassification Interpretation:
print('Misclassification Interpretation:')
for i in range(len(iris.target_names)):
    for j in range(len(iris.target_names)):
        if i!=j and cm_lr[i][j]>0:
            print(
                iris.target_names[i],
                'was predicted as:',
                iris.target_names[j],
                ':',
                cm_lr[i][j]
            )

# New Flower Prediction
sepal_length=float(input('Enter sepal length:'))
sepal_width=float(input('Enter sepal width:'))
petal_length=float(input('Enter petal length:'))
petal_width=float(input('Enter petal width:'))
new_flower=pd.DataFrame(
    [[sepal_length,sepal_width, petal_length, petal_width]],
    columns=iris.feature_names
)
predictions=model_lr.predict(new_flower)
print("Predicted Species:")
print(iris.target_names[predictions[0]])