import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def calculate_financial_risk_scores(financial_data_path: str) -> pd.DataFrame:
    df = pd.read_csv(financial_data_path)

    # Apply log1p transformation directly to 'Revenue (USD)'
    df['Revenue (USD)'] = np.log1p(df['Revenue (USD)'])

    # Select only numeric features (adjust as needed)
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    supplier_id_col = 'Supplier ID' if 'Supplier ID' in df.columns else df.columns[0]

    features = df[numeric_cols].copy()

    # Handle missing values
    features = features.fillna(features.mean())

    # Normalize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Isolation Forest
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(features_scaled)

    # Get anomaly scores (lower means more anomalous)
    anomaly_scores = -model.decision_function(features_scaled)

    # Scale to 0-100 (lower score = higher risk)
    from sklearn.preprocessing import MinMaxScaler
    scaled_scores = MinMaxScaler().fit_transform(anomaly_scores.reshape(-1, 1)) * 100
    df['Financial Risk Score'] = scaled_scores.round(2)

    return df[[supplier_id_col, 'Quarter', 'Financial Risk Score']]