# Import libraries
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the dataset
df = pd.read_csv("dataset-circle-toplology-1.csv", sep="\t", low_memory=False)

df.info()

df.isnull().sum()

# Drop unnecessary columns
df = df.drop(columns=['Sending_time'])

# Handle missing values - SimpleImputer
cat_imputer = SimpleImputer(strategy='most_frequent')
num_imputer = SimpleImputer(strategy='mean')
df[['rank', 'pdr_OF']] = num_imputer.fit_transform(df[['rank', 'pdr_OF']])

# Define features (X) and target variable (y)
X = df.drop(columns=['Energy_consumption'])  
y = df['Energy_consumption']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Define column transformer for preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply transformations
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Print
print("Preprocessing complete! Ready for model training.")

'''Decision Tree Regressor '''
from sklearn.tree import DecisionTreeRegressor
# Train the model
model_dt = DecisionTreeRegressor(
    max_depth=3,  
    min_samples_split=50,  
    min_samples_leaf=20,   
    random_state=42
)
model_dt.fit(X_train, y_train)
# Predictions
y_pred_dt = model_dt.predict(X_test)
# Evaluate Model
r2_dt = metrics.r2_score(y_test, y_pred_dt)
# Printing Evaluation Metric
print(f"Decision Tree R² Score: {r2_dt:.4f}")

'''Linear Regression'''
from sklearn.linear_model import LinearRegression
# Train the model
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
# Predictions
y_pred = model_lr.predict(X_test)
# Evaluate Model
r2_lr = metrics.r2_score(y_test, y_pred)
# Printing Evaluation Metric
print(f"Linear Regression R² Score: {r2_lr:.4f}")

'''Random Forest Regressor'''
from sklearn.ensemble import RandomForestRegressor
# Train the model
model_rf = RandomForestRegressor(n_estimators=100, max_features='sqrt', random_state=42)
model_rf.fit(X_train, y_train)
# Predictions
y_pred_rf = model_rf.predict(X_test)
# Evaluate Model
r2_rf = metrics.r2_score(y_test, y_pred_rf)
# Printing Evaluation Metric
print(f"Random Forest R² Score: {r2_rf:.4f}")

'''Support Vector Machine'''
from sklearn.svm import SVR
# Train the model
model_svr = SVR(kernel='linear', C=100, gamma='scale')
model_svr.fit(X_train, y_train)
# Predictions
y_pred_svr = model_svr.predict(X_test)
# Evaluate Model
r2_svr = metrics.r2_score(y_test, y_pred_svr)
# Printing Evaluation Metric
print(f"SVM R² Score: {r2_svr:.4f}")

'''Logistic Regression'''
from sklearn.linear_model import LogisticRegression

# Convert target variable into categorical (bins)
y_class = pd.qcut(y, q=3, labels=['Low', 'Medium', 'High'])  

# Split again for Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Train the Model
model_logreg = LogisticRegression(max_iter=500)
model_logreg.fit(X_train, y_train)

# Predictions
y_pred_logreg = model_logreg.predict(X_test)
# Evaluate Model
accuracy_logreg = metrics.accuracy_score(y_test, y_pred_logreg)
# Printing Evaluation Metric
print(f"Logistic Regression Accuracy: {accuracy_logreg:.4f}")

# Plotting comparison
import matplotlib.pyplot as plt

models = ['Linear Regression', 'Decision Tree', 'Random Forest', 'SVM', 'Logistic Regression']
scores = [r2_lr, r2_dt, r2_rf, r2_svr, accuracy_logreg]

plt.figure(figsize=(10, 6))
plt.bar(models, scores, color=['blue', 'green', 'red', 'purple', 'orange'])
plt.xlabel('Models')
plt.ylabel('Performance Score (R² or Accuracy)')
plt.title('Comparison of Model Performance')
plt.ylim(0, 1)
plt.show()




