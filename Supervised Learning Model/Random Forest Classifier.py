"""Random Forest: An ensemble learning method that combines multiple decision trees to improve accuracy and reduce overfitting."""

#import required libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns

#load the dataset
file_path = r"C:\Users\Admin\Desktop\AI - ML\student-mat.csv"  
#sep ; - the column seperated by an semicolon
#quotechar='"' = consider values in double quotes as sigle string
df = pd.read_csv(file_path, sep=';', quotechar='"')
print(df.head())

#Define Categorical Columns
categorical_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 
                    'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 
                    'nursery', 'higher', 'internet', 'romantic']

#label encoding - converting categorical to numerical - assign an numeric value to all datas in the categorical column(eg. 0,1,2,..) in an ordered way 
label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])  

#Create Target Variable(G3_pass)
df['G3_pass'] = (df['G3'] >= 10).astype(int)  #G3 final grade
# If G3 (final grade) is 10 or more, assign 1 (Pass).
# If G3 is less than 10, assign 0 (Fail).
#This converts a regression problem (predicting scores) into a classification problem (pass/fail prediction)

#Check Feature Correlation with Target
df_corr = df.drop(columns=['G3'])
correlation = df_corr.corr()['G3_pass'].drop('G3_pass').sort_values(ascending=False)
print("\nFeature Correlation with Target:")
print(correlation)
# Removes G3 from the dataset (since we now use G3_pass).
# Calculates correlation (.corr()) between features and the target variable (G3_pass).
# Sorts the correlation values in descending order.

#Define Features (X) and Target (y)
X = df.drop(columns=['G3', 'G3_pass'])  
y = df['G3_pass']  

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
#stratify=y: Ensures that the class distribution (pass/fail) is balanced in training and test sets

#Normalize Numeric Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Converts the scaled data back to a Pandas DataFrame
X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

#Train the RandomForest Model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

#display all features in the dataset
print("\nAll Features:")
print(X.columns.tolist())  

#Make Predictions & Evaluate Model
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")

#Feature Selection
#SelectFromModel: Helps in feature selection based on feature importance
#SelectFromModel() selects features whose importance is above the mean importance
selector = SelectFromModel(rf, threshold="mean")  
selector.fit(X_train, y_train)  

selected_features = X.columns[selector.get_support()]
print("\nSelected Features:", selected_features.tolist())

#Feature Selection Transformation
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

#Train a New RandomForest Model with Selected Features
rf_selected = RandomForestClassifier()              #Creates a new Random Forest classifier (rf_selected)
rf_selected.fit(X_train_selected, y_train)          #Trains it only on the selected features (X_train_selected)

#make prediction
y_pred = rf_selected.predict(X_test_selected)

#Trains it only on the selected features (X_train_selected)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

#Prints model performance metrics
print("\nOptimized Model Evaluation Metrics:")
print(f"1. Accuracy: {accuracy:.2f}")
print(f"2. Precision: {precision:.2f}")
print(f"3. Recall: {recall:.2f}")
print(f"4. F1-Score: {f1:.2f}")

#plot for selected features
feature_importances = rf_selected.feature_importances_

feat_importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': feature_importances})
feat_importance_df = feat_importance_df.sort_values(by='Importance', ascending=False)  

plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feat_importance_df, hue='Feature', palette="viridis", dodge=False, legend=False)

plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Student Performance Prediction")
plt.show()
# # ##

#correlation heatmap
plot_columns = X.columns.tolist() + ['G3_pass']
corr_matrix = df[plot_columns].corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.1f', linewidths=0.6)
plt.title('Correlation Heatmap')
plt.show()
#Displays correlations between all features and the target variable


