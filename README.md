# Heart Disease Prediction

A compact Streamlit app that predicts the likelihood of heart disease from a small set of clinical features. This project demonstrates data preprocessing, model inference, and a simple web UI for interactive predictions.

**Features:**
- Interactive Streamlit UI for entering patient features and getting a prediction
- Lightweight model inference in Python
- Clear instructions to set up and run locally

**Tech stack:** Python, Streamlit, scikit-learn, pandas, NumPy

## Quick start

Prerequisites:
- Python 3.8+ (use a virtual environment)

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser and use the form to enter patient data and see predictions.

## Dataset
The model was developed using public heart disease datasets (e.g., the UCI Heart Disease dataset - Cleveland). If you want to retrain or improve the model, download the UCI dataset and follow your preferred preprocessing and training workflow.

Dataset reference: https://archive.ics.uci.edu/ml/datasets/heart+Disease

## Project structure
- `app.py` — Streamlit application and inference code
- `requirements.txt` — Python dependencies

## Notes
- This repository is intended as a learning/demo project. For production use, add input validation, model versioning, unit tests, and secure deployment.

If you'd like, I can add a `train.py` script, example dataset download, or improve the UI flow next.
