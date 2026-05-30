import joblib

loan_model = joblib.load("models/loan_approval_model.pkl")
loan_scaler = joblib.load("models/loan_scaler.pkl")

segment_model = joblib.load("models/customer_segmentation_model.pkl")
segment_scaler = joblib.load("models/segmentation_scaler.pkl")

pca_model = joblib.load("models/pca_model.pkl")

print("All models loaded successfully!")