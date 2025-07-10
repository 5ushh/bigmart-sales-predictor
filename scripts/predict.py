import pandas as pd
import joblib
import json

# Load model
model = joblib.load('../models/sales_model.pkl')

# Load feature columns
with open('../models/feature_columns.json', 'r') as f:
    feature_columns = json.load(f)

# Input data dictionary (only include raw categorical values here)
input_data = {
    'Item_Identifier': ['FDA15'],
    'Item_Weight': [9.3],
    'Item_Fat_Content': ['Low Fat'],
    'Item_Visibility': [0.016047],
    'Item_Type': ['Household'],
    'Item_MRP': [249.8092],
    'Outlet_Identifier': ['OUT049'],
    'Outlet_Establishment_Year': [1999],
    'Outlet_Size': ['Medium'],
    'Outlet_Location_Type': ['Tier 1'],
    'Outlet_Type': ['Supermarket Type1']
}

# Create DataFrame
df = pd.DataFrame(input_data)

# Get dummies with the same columns as training
df_dummies = pd.get_dummies(df)

# Add missing dummy columns with zeros
for col in feature_columns:
    if col not in df_dummies.columns:
        df_dummies[col] = 0

# Ensure same column order
df_dummies = df_dummies[feature_columns]

# Predict
predicted_sales = model.predict(df_dummies)
print(f"🛒 Predicted Sales: {predicted_sales[0]:.2f}")
