import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam


# Prepare input features and target variable
X = stock_data[["MACD", "RSI", "MA5", "MA20", "MA60"]].values
y = stock_data["Close"].values

# Split data into training, validation, and test sets
X_train, X_val_test, y_train, y_val_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
X_val, X_test, y_val, y_test = train_test_split(
    X_val_test, y_val_test, test_size=0.5, shuffle=False
)

# Define the model architecture
model = Sequential(
    [
        LSTM(64, input_shape=(X_train.shape[1], 1), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1),
    ]
)

# Compile the model
model.compile(optimizer=Adam(), loss="mse")

# Train the model
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))

# Evaluate the model on the test set
loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
