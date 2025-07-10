import pandas as pd
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def train_model(df):
    # Drop rows with missing target (if any)
    df = df.dropna(subset=['Item_Outlet_Sales'])

    # Convert categorical columns using get_dummies
    df = pd.get_dummies(df, drop_first=True)

    # Separate features and target
    X = df.drop('Item_Outlet_Sales', axis=1)
    y = df['Item_Outlet_Sales']

    # Save feature columns for later
    feature_columns = list(X.columns)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print("✅ Model trained successfully!")
    print(f"📉 Mean Squared Error: {mse:.2f}")

    os.makedirs('models', exist_ok=True)
 
    model_path = os.path.abspath('models/sales_model.pkl')
    features_path = os.path.abspath('models/feature_columns.json')

    print(f"Saving model to: {model_path}")
    print(f"Saving features to: {features_path}")

    # Save model and features with error handling
    try:
        with open(model_path, 'wb') as f:
            joblib.dump(model, f)
        print("📦 Model saved successfully.")
    except Exception as e:
        print("❌ Error saving model:", e)

    try:
        with open(features_path, 'w') as f:
            json.dump(feature_columns, f)
        print("📦 Feature columns saved successfully.")
    except Exception as e:
        print("❌ Error saving feature columns:", e)

    print("Files currently in models directory:", os.listdir(os.path.abspath('../models')))


if __name__ == '__main__':
    # Load your data
    df = pd.read_csv('data/train.csv')

    # Train the model
    train_model(df)