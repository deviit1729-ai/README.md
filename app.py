from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = None


def load_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run train.py first.")
        model = joblib.load(MODEL_PATH)
    return model


@app.route("/", methods=["GET"]) 
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"]) 
def predict():
    try:
        quiz_score = float(request.form.get("quiz_score", 0))
        time_spent = float(request.form.get("time_spent", 0))
        assignments_completed = float(request.form.get("assignments_completed", 0))
        ai_used = int(request.form.get("ai_used", 0))

        X = pd.DataFrame([{
            "quiz_score": quiz_score,
            "time_spent": time_spent,
            "assignments_completed": assignments_completed,
            "ai_used": ai_used,
        }])

        clf = load_model()
        prob = clf.predict_proba(X)[0][1]
        pred = clf.predict(X)[0]
        label = "Pass" if pred == 1 else "Needs Support"

        return render_template("index.html", result=label, probability=round(float(prob), 4), inputs=X.iloc[0].to_dict())
    except Exception as e:
        return render_template("index.html", error=str(e))


if __name__ == "__main__":
    # Start the Flask app
    load_model()
    app.run(host="127.0.0.1", port=5000, debug=True)
