import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib


def generate_synthetic_data(n=2000, random_state=42):
    rng = np.random.RandomState(random_state)
    quiz_score = np.clip(rng.normal(70, 15, size=n), 0, 100)
    time_spent = np.clip(rng.normal(10, 4, size=n), 0, None)  # hours
    assignments_completed = np.clip(rng.beta(2, 1.5, size=n), 0, 1)  # fraction
    ai_used = rng.binomial(1, 0.2, size=n)

    # A simple hidden rule to determine passing
    latent = 0.5 * quiz_score + 4.5 * time_spent + 35 * assignments_completed + 8 * ai_used
    noise = rng.normal(0, 15, size=n)
    score = latent + noise

    passed = (score > 120).astype(int)

    df = pd.DataFrame({
        "quiz_score": quiz_score,
        "time_spent": time_spent,
        "assignments_completed": assignments_completed,
        "ai_used": ai_used,
        "passed": passed,
    })
    return df


def train_and_save_model(out_path="model.pkl"):
    df = generate_synthetic_data()
    X = df[["quiz_score", "time_spent", "assignments_completed", "ai_used"]]
    y = df["passed"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Logistic Regression pipeline
    log_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    log_pipe.fit(X_train, y_train)
    y_pred = log_pipe.predict(X_test)
    print("Logistic Regression accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Decision Tree (for comparison)
    dt = DecisionTreeClassifier(max_depth=6, random_state=1)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    print("Decision Tree accuracy:", accuracy_score(y_test, y_pred_dt))

    # Save the logistic pipeline as the production model
    joblib.dump(log_pipe, out_path)
    print(f"Saved trained model pipeline to {out_path}")


if __name__ == "__main__":
    train_and_save_model()
