
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

data = pd.read_csv("dataset.csv")

X = data.drop("label", axis=1).values
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, "ransomware_model.pkl")
print("Model trained and saved successfully")
