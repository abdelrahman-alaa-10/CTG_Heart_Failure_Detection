import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
# Load the dataset
dataset_path = 'fetal_health.csv'
data = pd.read_csv(dataset_path)

# Display the first few rows of the dataset
print(data.head())

# Split the data into features (X) and target (y)
X = data.drop(columns=['fetal_health'])
y = data['fetal_health']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict the target for the test set
y_pred = model.predict(X_test)
joblib.dump(model, "fetal_health_rf_model.pkl")

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

test_index = 0

print("***********************")
print(f"predicted class: {y_pred[test_index]}")
print(f"Actual Class: {y_test.iloc[test_index]}")