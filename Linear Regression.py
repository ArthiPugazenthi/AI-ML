"""Car Price Prediction""" 
"""Use a dataset containing car features (engine size, horsepower, age) to predict car prices. 
    Evaluate the model using RMSE and explain whether feature scaling improves performance. (linear)"""

"""Linear Regression: A regression model that predicts a continuous target variable by finding the best-fit linear relationship between features and output."""

#import relevent libraries 
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt

#load the dataset
df = pd.read_csv(r"C:\Users\Admin\Desktop\AI - ML\carpricedata1.csv")

#check a null value
df.isnull().sum()
#df.dropna(inplace=True) #to drop a null row

#is used to fill the missing value, object is filled by mode, [0]-first mode value and int,float is filled by median
df.fillna({
    'model_name': df['model_name'].mode()[0],
    'fuel': df['fuel'].mode()[0],
    'owner': df['owner'].mode()[0],
    'year': df['year'].median(),
    'km_driven': df['km_driven'].median(),
    'mileage': df['mileage'].median(),
    'seats': df['seats'].median()
}, inplace=True)

#print to check the null value
print(df.isnull().sum())

#data preprocessing

#split the dataset into features and target
X = df.iloc[:, :-1]  
y = df.iloc[:, -1]

#it list the object datatype
categorical_features = ['model_name', 'fuel', 'owner' ]
#it list the numeric datatype
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

#data transforming
#ColumnTransformer - apply different preproceesing to the different column
#OneHotEncoder - convert categorical to numeric like binary values(0,1)
#StandardScaler - scale them to mean=0, SD = 1, to get the same weightage for all column
#transformer - it transform standardscaler fot numeric and onehotencoder for categorical
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

print(df['model_name'].unique())


#apply the transformation to the X feature
#Fit: Learns the parameters (e.g., mean, standard deviation) of the transformers from the data.
#Transform: Applies these learned parameters to the data.
X_processed = preprocessor.fit_transform(X)

#spliting the data
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)

#model training
#a Linear Regression model is instantiated and trained using the training data (X_train, y_train).
model = LinearRegression()
model.fit(X_train, y_train)

#user input
model_name = input("Enter car model: ")
year = int(input("Enter year: ")) 
km_driven = float(input("Enter km_driven: "))
fuel = input("Enter fuel: ")
owner = input("Enter owner: ")
mileage = float(input("Enter mileage: "))
seats = float(input("Enter seats: "))

#user input values are stored in a new dataframe with a column name matching the original datasaet column name
user_input_df = pd.DataFrame([[model_name, year, km_driven, fuel, owner, mileage, seats]], columns=['model_name', 'year', 'km_driven', 'fuel', 'owner', 'mileage', 'seats'])

#The user input data is transformed using the same preprocessing pipeline (scaling and one-hot encoding) that was applied to the training data.
user_input_processed = preprocessor.transform(user_input_df)

#the transformed user input is passed to the model for prediction. The predicted price is printed to the user.
predicted_price = model.predict(user_input_processed)
print(f"\nThe predicted price of the car is: {predicted_price[0]:.2f}")  

#model evaluation
y_pred = model.predict(X_test)
#y_pred: The predicted car prices by the model.

rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))  #Root Mean Squared Error
mae = metrics.mean_absolute_error(y_test, y_pred)           #Mean Absolute Error
mse = metrics.mean_squared_error(y_test, y_pred)            #Mean Squared Error
r2 = metrics.r2_score(y_test, y_pred)                       #R-squared
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100    #Mean Absolute Percentage Error

print("\nModel Evaluation Metrics:")
print("RMSE:", rmse)
print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)
print("MAPE:", mape)

#plotting visualization
#Scatter Plot of Actual vs. Predicted Prices
plt.scatter(y_test, y_pred)    #y_test: The actual car prices from the test dataset.
plt.xlabel("Actual Prices")     #True values from the dataset.
plt.ylabel("Predicted Prices")  #Model's predicted values.
plt.title("Actual vs Predicted Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2) 
#y_test.min() and y_test.max(): Get the minimum and maximum values from actual prices.
#[y_test.min(), y_test.max()]-the line extends from the lowest price to the highest price

#Highlighting User's Car Price Prediction
plt.scatter(predicted_price, predicted_price, color='red', marker='*', s=100, label="User Input Prediction")
plt.show()  



