import joblib
from utils.dataset import load_dataset
from utils.preprocessing import preprocess_data
from utils.training import train_model 
from utils.constants import CLEANED_DATASET
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

# Load the dataset
data = load_dataset(CLEANED_DATASET)

# Preprocess the dataset
drop_columns = ['Unnamed: 0', 'flight']
scaler_path = 'models/preprocessing/gbr_scaler.pkl'
encoder_path = 'models/preprocessing/gbr_encoder.pkl'
data, scaler, encoder = preprocess_data(data, drop_columns, scaler_path, encoder_path)

# Variables
X = data.drop(columns=['price'])
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model and evaluate
model = GradientBoostingRegressor()
trained_model, mae, r2 = train_model(model, X_train, y_train, X_test, y_test)

# Save the trained model
model_path = 'models/'
model_filename = 'gbr_model.pkl'
joblib.dump(trained_model, model_path + model_filename)

# Preview the results
print(f'Model Name: {model.__class__.__name__}')
print(f'Mean Absolute Error: {mae}')
print(f'R2 Score: {r2}\n')
print(f'Model saved at: {model_path + model_filename}')