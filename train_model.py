
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the crop dataset
data = pd.read_csv("data/Crop_recommendation.csv")

# Input features
X = data[["N", "P", "K", "temperature",
          "humidity", "ph", "rainfall"]]

# Output crop name
y = data["label"]

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create AI models
decision_tree = DecisionTreeClassifier(random_state=42)

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train both models
decision_tree.fit(X_train, y_train)
random_forest.fit(X_train, y_train)

# Test both models
dt_predictions = decision_tree.predict(X_test)
rf_predictions = random_forest.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_predictions)
rf_accuracy = accuracy_score(y_test, rf_predictions)

print("Decision Tree Accuracy:",
      round(dt_accuracy * 100, 2), "%")

print("Random Forest Accuracy:",
      round(rf_accuracy * 100, 2), "%")

# Select the better model based on test accuracy
if rf_accuracy >= dt_accuracy:
    best_model = random_forest
    best_name = "Random Forest"
else:
    best_model = decision_tree
    best_name = "Decision Tree"

# Save the selected model
os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/crop_model.pkl")

# Save feature names for explanations
print("\nSelected Model:", best_name)
print("Model saved successfully!")
print("\nImportant features:")

for feature, importance in sorted(
    zip(X.columns, best_model.feature_importances_),
    key=lambda item: item[1],
    reverse=True
):
    print(feature, ":", round(importance, 3))
