import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

def generate_supplier_score(
    overview_path: str,
    financial_score_path: str,
    master_data_path: str,
    financial_data_path: str,
    output_path: str
) -> None:
    # Load all datasets
    overview_df = pd.read_csv(overview_path)
    financial_df = pd.read_csv(financial_score_path)
    master_df = pd.read_csv(master_data_path)
    raw_financial_df = pd.read_csv(financial_data_path)
    
    
    # Clean up column names
    overview_df.columns = overview_df.columns.str.strip()
    financial_df.columns = financial_df.columns.str.strip()
    master_df.columns = master_df.columns.str.strip()
    raw_financial_df.columns = raw_financial_df.columns.str.strip()
    
    # Rename columns for standardization
    overview_df.rename(columns={"Qtr": "Quarter", "quarter": "Quarter"}, inplace=True)
    financial_df.rename(columns={"Qtr": "Quarter", "quarter": "Quarter"}, inplace=True)
    raw_financial_df.rename(columns={"Qtr": "Quarter", "quarter": "Quarter"}, inplace=True)
    
    # Only keep relevant columns
    overview_cols = ["Supplier ID", "Quarter", "%_on_time", "%_delayed", "avg_delay",
                     "avg_shipment_volume", "%_shipment_lost", "%_defect_rate"]
    financial_cols = ["Supplier ID", "Quarter", "Financial Risk Score"]
    master_cols = ["Supplier ID", "Tier"]
    raw_financial_cols = ["Supplier ID", "Quarter", "Credit Score", "Revenue (USD)", "D&B Rating"]
    overview_df = overview_df[overview_cols]
    financial_df = financial_df[financial_cols]
    master_df = master_df[master_cols]
    raw_financial_df = raw_financial_df[raw_financial_cols]
    
    # Merge datasets on both Supplier ID and Quarter
    merged_df = overview_df.merge(financial_df, on=["Supplier ID", "Quarter"], how="inner")
    merged_df = merged_df.merge(raw_financial_df, on=["Supplier ID", "Quarter"], how="left")
    merged_df = merged_df.merge(master_df, on="Supplier ID", how="left")
    
    # Feature engineering
    merged_df["Tier"] = LabelEncoder().fit_transform(merged_df["Tier"].astype(str))
    merged_df["delay_defect_interaction"] = merged_df["%_defect_rate"] * merged_df["%_delayed"]
    merged_df["volume_delay_interaction"] = merged_df["avg_shipment_volume"] * merged_df["avg_delay"]
    merged_df["volume_lost_interaction"] = merged_df["avg_shipment_volume"] * merged_df["%_shipment_lost"]
    merged_df["defect_lost_interaction"] = merged_df["%_defect_rate"] * merged_df["%_shipment_lost"]
    merged_df["log_avg_delay"] = np.log1p(merged_df["avg_delay"])
    merged_df["log_shipment_volume"] = np.log1p(merged_df["avg_shipment_volume"])
    merged_df["Log Revenue"] = np.log1p(merged_df["Revenue (USD)"])
    
    # Features and dummy target
    feature_cols = [
        "%_on_time", "%_delayed", "avg_delay", "avg_shipment_volume",
        "%_shipment_lost", "%_defect_rate", "Financial Risk Score", "Tier",
        "delay_defect_interaction", "volume_delay_interaction",
        "volume_lost_interaction", "defect_lost_interaction",
        "log_avg_delay", "log_shipment_volume",
        "Credit Score", "D&B Rating", "Log Revenue"
    ]
    X = merged_df[feature_cols]
    y = (
        0.2256 * (100 - merged_df["%_shipment_lost"]) +
        0.188 * (100 - merged_df["%_defect_rate"]) +
        0.1504 * (100 - merged_df["%_delayed"]) +
        0.1353 * (100 - merged_df["avg_delay"]) +
        0.0375 * (100 - merged_df["Financial Risk Score"]) +
        0.1128 * (merged_df["Credit Score"]) +
        0.0752 * (100 - merged_df["D&B Rating"]) +
        0.0752 * (100 - merged_df["Log Revenue"])
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train/test split and model fitting
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = XGBRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Predict scores for all data
    merged_df["Supplier Score"] = model.predict(scaler.transform(X))

    # Rescale Supplier Score to 0-100 range
    minmax_scaler = MinMaxScaler(feature_range=(0, 100))
    merged_df["Supplier Score"] = minmax_scaler.fit_transform(merged_df[["Supplier Score"]])

    # GROUP BY Supplier ID + Quarter, average if duplicates
    result = merged_df.groupby(["Supplier ID", "Quarter"], as_index=False)["Supplier Score"].mean()

    # Save final dataset
    result.to_csv(output_path, index=False)
    print(f"Supplier scores saved to {output_path}")