import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import joblib

#Load a housing Dataset:
df=pd.read_csv('houses.csv')
print(df)

print('First 5 rows:',df.head())

print('\nNumber of rows:',df.shape[0])
print('Number of columns:',df.shape[1])
print('Shape is:',df.shape)


#Explore Dataset:
print('\nData set information:',df.info())
print('\nStatistical Summary is:',df.describe())

#Data cleaning:
print('\nMissing values:',df.isnull().sum())

print('Duplicated rows :',df.duplicated().sum())
print(df.drop_duplicates())


print('Data after removing duplicated rows:',df.shape)

#Data visualization:
sns.scatterplot(x=df['Area'],y=df['Price'])
plt.title('House price prediction')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()



#FEATURES AND TARGET:
x=df[['Area','Bedrooms','Bathrooms']]
y=df['Price']
print('\nFeatures',x)
print('Target',y)

#Train/Test Split:

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=10)
print('\nTraining features shape:', x_train.shape)
print('Testing features shape:',x_test.shape)
print('Traing target shape:',y_train.shape)
print('Testing target shape:',y_test.shape)

#Linear Regression Model:

model=LinearRegression()
model.fit(x_train,y_train)
print('\nModel trained Successfully')

#Predictions:

predictions=model.predict(x_test)
print('\nActual Prices')
print(y_test.values)
print('\nPredicted Prices')
print(predictions)

#Model Evaluation:

rmse=mean_squared_error(y_test,predictions)**0.5
r2=r2_score(y_test,predictions)
print('RMSE:',rmse)
print('R2_Score',r2)

#Model Coefficients:
print('Area:',model.coef_[0])
print('Bedrooms:',model.coef_[1])
print('Bathrooms:',model.coef_[2])
print('Model Coefficients:')
print('Intercept:',model.intercept_)


#Save Model:

joblib.dump(model,'house_price_model.pkl')

# Example:
new_house=pd.DataFrame({
    "Area": [2000],
    "Bedrooms": [4],
    "Bathrooms": [3]
})
predicted_price=model.predict(new_house)
print('\nExample Predictions:')
print('Predicted Price:',predicted_price[0])















