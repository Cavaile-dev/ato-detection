from __future__ import annotations
import pandas as pd
from server.config import DATA_DIR, FEATURE_COLUMNS, MOCK_FEATURE_PATH
REAL_FEATURES_PATH = DATA_DIR / "real_features.csv"
def bootstrap_training_data() -> pd.DataFrame:
    """
    Load training data.
    
    Priority:
        1. Real data (real_features.csv) if exists and has >= 10 rows
        2. Mock synthetic data (mock_features.csv)
        3. Generate new mock data
    """
    if REAL_FEATURES_PATH.exists():
        real_df = pd.read_csv(REAL_FEATURES_PATH)
        if len(real_df) >= 10:
            print(f"[train_model] Using REAL data: {len(real_df)} sessions")
            return real_df
    
    if MOCK_FEATURE_PATH.exists():
        existing = pd.read_csv(MOCK_FEATURE_PATH)
        if len(existing) >= 100:
            print(f"[train_model] Using MOCK data: {len(existing)} sessions")
            return existing
    
    from server.mock_data import generate_mock_dataset
    print("[train_model] No data found, generating mock...")
    frame, _ = generate_mock_dataset()
    return frame
def train_default_model() -> dict:
    """Train the anomaly model using available data."""
    feature_frame = bootstrap_training_data()
    
    normal_frame = feature_frame[feature_frame["label"] == 0][FEATURE_COLUMNS]
    
    print(f"[train_model] Training on {len(normal_frame)} normal sessions")
    
    from server.model import BehavioralAnomalyModel
    model_service = BehavioralAnomalyModel()
    return model_service.train(normal_frame)

def train_with_real_data() -> dict:
    """Force collect real data and train."""
    from server.real_data_collector import collect_real_training_data
    
    df = collect_real_training_data()
    
    if df is None:
        raise ValueError("Not enough real sessions to train. Collect more data first.")
    
    normal_df = df[df["label"] == 0][FEATURE_COLUMNS]
    
    from server.model import BehavioralAnomalyModel
    model_service = BehavioralAnomalyModel()
    return model_service.train(normal_df)

def main():
    summary = train_default_model()
    print("\nModel trained successfully!")
    for key, value in summary.items():
        print(f"  {key}: {value}")
        
if __name__ == "__main__":
    main()