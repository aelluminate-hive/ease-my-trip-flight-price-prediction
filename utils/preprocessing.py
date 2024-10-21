import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def preprocess_data(data, drop_columns, scaler_path=None, encoder_path=None):
    # Drop specified columns
    data = data.drop(columns=drop_columns)
    
    # Separate the numerical and categorical columns
    numerical = data.select_dtypes(include=np.number).columns
    categorical = data.select_dtypes(include='object').columns
    
    # Scale the numerical columns
    scaler = MinMaxScaler()
    data[numerical] = scaler.fit_transform(data[numerical])
    
    # One-hot encode the categorical columns
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(data[categorical])
    encoder_cols = encoder.get_feature_names_out(categorical)
    data[encoder_cols] = encoder.transform(data[categorical])
    
    # Drop original categorical columns
    data = data.drop(columns=categorical)
    
    # Save the scaler and encoder if paths are provided
    if scaler_path:
        joblib.dump(scaler, scaler_path)
    if encoder_path:
        joblib.dump(encoder, encoder_path)
    
    return data, scaler, encoder