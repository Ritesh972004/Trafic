# train_save_model.py
"""
Creates a toy scikit-learn model and saves it to model/traffic_model.pkl
Run: python train_save_model.py
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import joblib
from pathlib import Path

def create_and_save_model(out_path: Path = Path("model") / "traffic_model.pkl"):
    X, y = make_classification(
        n_samples=400,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        random_state=42
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_path)
    print(f"Saved sample model to {out_path}")

if __name__ == "__main__":
    create_and_save_model()
