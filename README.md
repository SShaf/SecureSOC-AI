# SecureSOC AI

A SOC simulation platform built as part of a Final Year Project at Universiti Teknologi Brunei (UTB) in collaboration with Multimedia University (MMU) Cyberjaya.

## Project Title
A Deep Learning-Based Intrusion Detection System for False Positive Reduction and SOC Analyst Workload Optimisation

## Author
Mohammad Shahrul Shafie Bin Shukri (B20230382)
BSc (Hons) Information Security, UTB

## What This Platform Does
- Replays CICIDS2017 validation traffic through both Random Forest and DNN-BiLSTM-Attention models in real time
- Shows live alert queue with both model verdicts, attack type, agreement, and whitelist result
- SHAP explainability per alert
- Inject mode for controlled attack demonstrations
- Model comparison page with labelled and raw unlabelled dataset results
- Automated PDF report generator

## Tech Stack
- Backend: Flask (Python)
- Frontend: HTML, CSS, JavaScript
- ML: scikit-learn (Random Forest), PyTorch (DNN-BiLSTM-Attention)
- Explainability: SHAP TreeExplainer

## How to Run
1. Install dependencies: pip install flask numpy pandas scikit-learn torch shap fpdf2 joblib
2. Place model files in models/ folder (rf_model.pkl, bilstm_best_full.pt, scaler.pkl, whitelist_profile.csv, ymc_val.npy)
3. Place data files in data/ folder (X_val.npy, y_val.npy, feature_names.txt)
4. Run: python app.py
5. Open http://127.0.0.1:5000

## Note
Model and data files are not included in this repository due to file size. Contact the author for access.
