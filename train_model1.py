import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the simulated sensor dataset
df = pd.read_csv("sensor_data.csv")

# Create output folder
os.makedirs("outputs", exist_ok=True)

X = df[[
    "temperature",
    "vibration",
    "pressure",
    "operating_hours",
]]
y = df["failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Model training completed!")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, "outputs/predictive_maintenance_model.pkl")
print("\nModel saved as outputs/predictive_maintenance_model.pkl")
print("Files saved:", os.listdir("outputs"))
