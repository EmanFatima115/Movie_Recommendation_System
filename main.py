import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df=pd.read_csv('houses.csv')
print('First 5 rows:',df.head())
print('No of rows:',df.shape[0])
print('shape:',df.shape)
print(df.describe())
sns.scatterplot(x=df['Area'],y=df['Price'])
plt.show()

x=df[['Area','Bedrooms','Bathrooms']]
y=df['Price']
print('Features are:',x)
print("Target is:",y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=10)
print(x_train.shape)
print(y_train.shape)
print('x_test',x_test)
print('y_test',y_test)
model=LinearRegression()
model.fit(x_train,y_train)
predictions=model.predict(x_test)

print('Actual Prices')
print(y_test)
print('Predicted prices')
print(predictions)








