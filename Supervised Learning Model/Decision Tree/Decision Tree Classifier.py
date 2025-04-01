"""Weather Prediction 
Use a dataset with weather conditions (temperature, humidity, wind) to predict 
whether it will rain. Perform hyperparameter tuning using grid search. (DT)""" 

"""Decision Tree: A tree-based model that splits data into branches based on feature conditions, making decisions in a stepwise manner."""

import numpy as np   #Used for numerical operations.
import pandas as pd  # Used for handling datasets in tabular format.
from sklearn.model_selection import train_test_split, GridSearchCV
         # train_test_split: Splits the data into training and testing datasets.
         # GridSearchCV - Used for hyperparameter tuning (finding the best model parameters).
from sklearn.tree import DecisionTreeClassifier       #A machine learning model used for classification tasks.
from sklearn.metrics import accuracy_score, precision_score, recall_score    #metrics to evaluate

#load the dataset
df = pd.read_csv(r"C:\Users\Admin\Desktop\AI - ML\weather.csv")

#classify feature and target
X = df.iloc[:, :-1]  #extract all column except last one
y = df.iloc[:, -1]   #extract only last column

#split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#This is used for evaluating the model's performance on unseen data and avoiding overfitting. 
#for training 80% and testing 20%
#random_state - When you set random_state to a specific number (like 42), you'll get the same results every time you run the code, which is useful for consistency and debugging.

#Initialize the decision tree
model = DecisionTreeClassifier()
#It can handle both numerical and categorical data and automatically performs feature selection.
#It works by splitting the data based on feature values to predict the target variable.

#a dictionary that contains the hyperparameters for the DTC that we want to tune
param_grid = {
    'max_depth': [3, 5, 10, None],       #This parameter controls the maximum depth of the decision tree. it will stop the splitting when it reach 3,5,10
    'min_samples_split': [2, 5, 10],     #This parameter specifies the minimum number of samples required to split an internal node.
    'min_samples_leaf': [1, 2, 4]        #This parameter specifies the minimum number of samples required to be at a leaf node
}

#It evaluates the performance of different hyperparameter combinations by training the model on the training data and testing it using cross-validation.
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train) #it will train the model

#This is a fully trained model that you can now use to make predictions on new data, such as on the test set.
best_model = grid_search.best_estimator_

#The model's predictions on the test data (X_test) are stored in y_pred.
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)       #The proportion of correct predictions out of all predictions.
precision = precision_score(y_test, y_pred)     #The proportion of positive predictions that were actually correct.
recall = recall_score(y_test, y_pred)           #The proportion of true positive predictions among all actual positives.

#print the metrics
print("\nModel Evaluation Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

#user input
temp = float(input("\nEnter Temperature (°C): "))
humidity = float(input("Enter Humidity (%): "))
wind_speed = float(input("Enter Wind Speed (km/h): "))

prediction = best_model.predict([[temp, humidity, wind_speed]])
result = "It will rain!" if prediction[0] == 1 else "No rain expected."

print(result)

