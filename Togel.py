import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# 1. Persiapan Data
data = pd.read_csv('data_lottery.csv')
numbers = data['Numbers']
results = data['Result']

# 2. Pembagian Data
numbers_encoded = OneHotEncoder().fit_transform(numbers.values.reshape(-1, 1))
X_train, X_val, y_train, y_val = train_test_split(numbers_encoded, results, test_size=0.2, random_state=42)

# 3. Pembangunan Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(len(numbers_encoded[0]),)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Pelatihan Model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# 5. Evaluasi Model
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_accuracy)

# 6. Prediksi
new_data = np.array([[2, 4, 6, 8, 10]])
prediction = model.predict(new_data)
print("Prediction:", prediction)
