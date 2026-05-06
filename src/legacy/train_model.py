from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

def add_features(df):
    if 'Close' not in df.columns:
        raise KeyError(f"'Close' column not found. Columns: {df.columns.tolist()}")

    df['Return'] = df['Close'].pct_change()
    df['Target'] = (df['Return'].shift(-1) > 0).astype(int)
    df.dropna(inplace=True)
    return df

def train_base_model(df):
    """
    Train a basic Random Forest classifier.
    """
    features = ['Return']
    X = df[features]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False, test_size=0.2
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("Training complete. Evaluation:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    return model