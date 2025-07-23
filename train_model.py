import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from xgboost import XGBClassifier

# Step 1: Create dummy data with 64 features
X = np.random.rand(100, 64)
y = np.random.randint(0, 2, size=(100,))

# Step 2: Split and scale
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ✅ Save the scaler
joblib.dump(scaler, "scaler.pkl")

# Step 3: Train ANN model
ann_model = Sequential()
ann_model.add(Dense(64, input_dim=64, activation='relu'))
ann_model.add(Dense(1, activation='sigmoid'))
ann_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
ann_model.fit(X_train_scaled, y_train, epochs=10, batch_size=16, verbose=1)

# ✅ Save ANN model
ann_model.save("ann_model.keras")

# Step 4: Train XGBoost model (NO use_label_encoder)
xgb_model = XGBClassifier(eval_metric='logloss')
xgb_model.fit(X_train_scaled, y_train)

# ✅ Save XGB model using joblib
joblib.dump(xgb_model, "xgb_model.pkl")

print("✅ ANN model, XGBoost model, and scaler saved successfully.")



