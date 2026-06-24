import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

class DNNEncoder(nn.Module):
    def __init__(self, input_dim=77, dropout=0.3):
        super(DNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.classifier(x)
        return self.sigmoid(x)

class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(AdditiveAttention, self).__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim // 2, bias=False)
        self.v = nn.Linear(hidden_dim // 2, 1, bias=False)

    def forward(self, lstm_output):
        energy = torch.tanh(self.W(lstm_output))
        attn_weights = torch.softmax(self.v(energy), dim=1)
        context = torch.sum(attn_weights * lstm_output, dim=1)
        return context, attn_weights

class DNNBiLSTMAttention(nn.Module):
    def __init__(self, input_dim=77, dnn_hidden=64, lstm_hidden=64, dropout=0.3):
        super(DNNBiLSTMAttention, self).__init__()
        self.dnn_encoder = DNNEncoder(input_dim, dropout)
        self.bilstm = nn.LSTM(dnn_hidden, lstm_hidden, batch_first=True, bidirectional=True)
        self.attention = AdditiveAttention(lstm_hidden * 2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dnn_encoder.layer1(x)
        x = self.dnn_encoder.layer2(x)
        x = x.unsqueeze(1)
        lstm_out, _ = self.bilstm(x)
        context, attn_weights = self.attention(lstm_out)
        context = self.dropout(context)
        out = self.sigmoid(self.classifier(context))
        return out

# 77 raw feature names, in the same order the scaler/model expect
with open('data/feature_names.txt') as f:
    FEATURE_NAMES = [line.strip() for line in f if line.strip()]


def predict_bilstm(row, scaler, bilstm_model, whitelist_df):
    row = np.asarray(row, dtype=np.float32).reshape(1, -1)
    scaled = scaler.transform(row)

    bilstm_model.eval()
    with torch.no_grad():
        prob = bilstm_model(torch.tensor(scaled, dtype=torch.float32)).item()

    prediction = 1 if prob >= 0.5 else 0
    confidence = prob if prediction == 1 else 1 - prob

    whitelist_result = 'N/A'
    if prediction == 1:
        whitelist_result = 'Confirmed'
        for feature, bounds in whitelist_df.iterrows():
            idx = FEATURE_NAMES.index(feature)
            value = scaled[0, idx]
            if bounds['lower'] <= value <= bounds['upper']:
                whitelist_result = 'Suppressed'
                prediction = 0
                break

    return prediction, confidence, whitelist_result


if __name__ == '__main__':
    scaler = joblib.load('models/scaler.pkl')

    bilstm_model = torch.load('models/bilstm_best_full.pt', map_location='cpu', weights_only=False)
    bilstm_model.eval()

    whitelist_df = pd.read_csv('models/whitelist_profile.csv', index_col=0)

    X_val = np.load('data/X_val.npy')
    y_val = np.load('data/y_val.npy')

    # X_val is already scaled, so invert it to get a raw feature row to test with
    raw_row = scaler.inverse_transform(X_val[:1])[0]

    prediction, confidence, whitelist_result = predict_bilstm(raw_row, scaler, bilstm_model, whitelist_df)

    print("True label:", "Attack" if y_val[0] == 1 else "BENIGN")
    print("Prediction:", "Attack" if prediction == 1 else "BENIGN")
    print("Confidence:", round(confidence, 4))
    print("Whitelist result:", whitelist_result)
