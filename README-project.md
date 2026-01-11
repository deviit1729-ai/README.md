Student Pass Predictor

This small project trains a classifier to predict whether a student will `Pass` or `Needs Support` based on:
- Quiz scores
- Time spent on lessons
- Assignment completion
- AI used (binary)

Quickstart

1. Create a virtual environment and install dependencies:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Train the model (creates `model.pkl`):

```powershell
python train.py
```

3. Run the Flask app:

```powershell
python app.py
```

4. Open http://127.0.0.1:5000 in your browser and test the form.

Notes
- `train.py` generates synthetic data if you don't have a dataset. Replace data generation with CSV loading if you have real data.
- The production model saved is a Logistic Regression pipeline (`model.pkl`).
