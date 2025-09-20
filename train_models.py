
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.svm import SVC
import os

# Load dataset
df = pd.read_csv("dataset.csv")

# Features and labels
X = df.drop("label", axis=1)  # adjust "label" to match your dataset column
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Logistic Regression": pickle.load(open("models/logistic_regression.pkl", "rb")),
    "KNN": pickle.load(open("models/knn.pkl", "rb")),
    "Decision Tree": pickle.load(open("models/decision_tree.pkl", "rb")),
    "Random Forest": pickle.load(open("models/random_forest.pkl", "rb")),
    "Bagging": pickle.load(open("models/bagging.pkl", "rb")),
    "AdaBoost": pickle.load(open("models/adaboost.pkl", "rb")),
    "Gradient Boosting": pickle.load(open("models/gradient_boosting.pkl", "rb")),
    "Extra Trees": pickle.load(open("models/extra_trees.pkl", "rb")),
    "SVM": pickle.load(open("models/svm.pkl", "rb"))
}


# Make directory for models
os.makedirs("models", exist_ok=True)

# Train & Save models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy : {acc :.2f}")
    
    with open(f"models/{name.lower().replace(' ', '_')}.pkl", "wb") as f:
        pickle.dump(model, f)

print("✅ All models trained & saved in 'models/' folder")
