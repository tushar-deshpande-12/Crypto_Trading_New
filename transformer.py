
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MultiHeadAttention, Dense, Dropout, LayerNormalization, GlobalAveragePooling1D, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def build_transformer_model(input_shape, symbol):
    # Define input layer
    inputs = Input(shape=input_shape)
    
    # Add MultiHeadAttention layer
    attention_output = MultiHeadAttention(num_heads=8, key_dim=64)(inputs, inputs)
    
    # Add Layer Normalization
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
    
    # Add residual connection
    attention_output = Add()([inputs, attention_output])
    
    # Add another MultiHeadAttention layer
    attention_output = MultiHeadAttention(num_heads=8, key_dim=64)(attention_output, attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
    attention_output = Add()([inputs, attention_output])
    
    # Add Global Average Pooling
    pooled_output = GlobalAveragePooling1D()(attention_output)
    
    # Add Dense and Dropout layers
    x = Dense(512, activation='relu', kernel_regularizer='l2')(pooled_output)
    x = Dropout(0.2)(x)
    
    x = Dense(256, activation='relu', kernel_regularizer='l2')(x)
    x = Dropout(0.2)(x)

    x = Dense(128, activation='relu', kernel_regularizer='l2')(x)
    x = Dropout(0.2)(x)
    
    # Output layer
    outputs = Dense(1)(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model with a learning rate scheduler
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    # Save the model
    model_filename = f'transformer_model_{symbol}.h5'
    model.save(model_filename)
    print(f"Model saved as {model_filename}")
    
    return model

def create_sequences(data, seq_length=20):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i+seq_length].values)
        y.append(data.iloc[i+seq_length]['Close'])
    return np.array(X), np.array(y)

def calculate_time_to_profit(investment_time, profitable_time):
    """
    Calculate the time difference between investment and profitability.

    Parameters:
    - investment_time (datetime): The time of investment.
    - profitable_time (str or datetime): The time when the investment becomes profitable.

    Returns:
    - str: A string describing the time to profit or 'Never becomes profitable'.
    """
    if isinstance(profitable_time, str) and profitable_time == "Never becomes profitable":
        return "Never becomes profitable"

    if isinstance(profitable_time, str):
        try:
            profitable_time = datetime.strptime(profitable_time, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return "Error: Invalid profitable_time format"

    if not isinstance(investment_time, datetime):
        try:
            investment_time = datetime.strptime(investment_time, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return "Error: Invalid investment_time format"

    if profitable_time <= investment_time:
        return "Already profitable at investment time"

    time_difference = profitable_time - investment_time
    
    days = time_difference.days
    hours, remainder = divmod(time_difference.seconds, 3600)
    minutes, _ = divmod(remainder, 60)

    result = []
    if days > 0:
        result.append(f"{days} day{'s' if days > 1 else ''}")
    if hours > 0:
        result.append(f"{hours} hour{'s' if hours > 1 else ''}")
    if minutes > 0:
        result.append(f"{minutes} minute{'s' if minutes > 1 else ''}")

    return "Time to profit: " + ", ".join(result)

def train_transformer_model(data, symbol, seq_length=20, epochs= 120, batch_size= 10):
    # Use multiple features
    feature_columns = ['Close', 'SMA_5', 'SMA_20', 'SMA_100', 'EMA_5', 'EMA_20', 'EMA_100', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9', 'RSI_14']
    scaled_data = data[feature_columns]
    
    # Create sequences
    X, y = create_sequences(scaled_data, seq_length)
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Build the model
    model = build_transformer_model((X_train.shape[1], X_train.shape[2]), symbol)
    
    # Learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[lr_scheduler])
    
    # Evaluate the model
    loss = model.evaluate(X_val, y_val)
    print(f'Validation Loss: {loss}')
    
    # Plot training & validation loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()
    
    return model, X_val, y_val

def evaluate_model(model, X_val, y_val):
    """
    Evaluate the model's performance on the validation set and plot the actual vs. predicted values.

    Parameters:
    - model: Trained transformer model.
    - X_val (np.array): Validation input data.
    - y_val (np.array): Validation target data.

    Returns:
    - float: Validation loss.
    """
    # Predict the validation set
    y_pred = model.predict(X_val)
    
    # Calculate the validation loss
    loss = mean_squared_error(y_val, y_pred)
    print(f'Validation Loss: {loss}')
    
    # Calculate the prediction errors (residuals)
    prediction_errors = y_val - y_pred.flatten()
    
    # Calculate the standard deviation of the prediction errors
    std_dev = np.std(prediction_errors)
    print(f'Standard Deviation of Prediction Errors: {std_dev}')
    
    # Calculate upper and lower bounds
    upper_bound = y_pred.flatten() + std_dev
    lower_bound = y_pred.flatten() - std_dev
    
    # Plot actual vs. predicted values
    plt.figure(figsize=(14, 7))
    plt.plot(y_val, label='Actual Values', color='blue')
    plt.plot(y_pred, label='Predicted Values', color='orange', linestyle='--')
    plt.fill_between(range(len(y_val)), lower_bound, upper_bound, color='gray', alpha=0.2, label='Prediction Std Dev')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Actual vs. Predicted Values with Standard Deviation Bounds')
    plt.legend()
    plt.show()
    
    return loss

def predict_output_transformer(model, data, buy_price_usdt, investment_time, look_back=20, num_future_steps=30):
    """
    Predict future cryptocurrency prices using a trained transformer model and calculate the standard deviation of prediction errors.
    """
    # Ensure 'Open time' is in datetime format
    data['Open time'] = pd.to_datetime(data['Open time'])

    # Use multiple features, excluding 'Open time'
    feature_columns = ['Close', 'SMA_5', 'SMA_20', 'SMA_100', 'EMA_5', 'EMA_20', 'EMA_100', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9', 'RSI_14']
    numeric_data = data[feature_columns].astype(float)

    # Initialize the input array (X_input) for prediction
    X_input = numeric_data.iloc[-look_back:].values.reshape(1, look_back, -1)

    # Ensure the data is of type float32
    X_input = X_input.astype(np.float32)

    # List to store future predictions
    future_predictions = []

    # Variable to store the time when the trade becomes profitable
    profitable_time = "Never becomes profitable"

    # Get the last date from the original data
    last_date = data['Open time'].iloc[-1]

    for step in range(num_future_steps):
        # Predict the next value
        prediction = model.predict(X_input)
        
        # Store the prediction
        future_predictions.append(prediction[0, 0])
        
        # Check if the trade becomes profitable
        if prediction[0, 0] > buy_price_usdt and profitable_time == "Never becomes profitable":
            profitable_time = last_date + timedelta(hours=4*(step+1))
        
        # Update the input data
        new_row = np.copy(X_input[0, -1, :])
        new_row[0] = prediction[0, 0]  # Update the 'Close' value
        X_input = np.roll(X_input, -1, axis=1)
        X_input[0, -1, :] = new_row

    # Prepare future dates for plotting
    future_dates = pd.date_range(start=last_date, periods=num_future_steps + 1, freq='4H')[1:]
    predictions_df = pd.DataFrame({'Open time': future_dates, 'Predicted Close': future_predictions})

    # Calculate the prediction errors (residuals)
    actual_values = data['Close'].iloc[-num_future_steps:].values
    prediction_errors = actual_values - np.array(future_predictions[:len(actual_values)])
    
    # Calculate the standard deviation of the prediction errors
    std_dev = np.std(prediction_errors)

    # Calculate upper and lower bounds
    predictions_df['Upper Bound'] = predictions_df['Predicted Close'] + std_dev
    predictions_df['Lower Bound'] = predictions_df['Predicted Close'] - std_dev

    # Check if predictions are within a reasonable range
    print(predictions_df.describe())
    print(f'Standard Deviation of Prediction Errors: {std_dev}')

    # Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(data['Open time'], data['Close'], label='Actual Close Prices', color='blue')
    plt.plot(predictions_df['Open time'], predictions_df['Predicted Close'], label='Predicted Future Prices', color='orange', linestyle='--')
    plt.fill_between(predictions_df['Open time'], predictions_df['Lower Bound'], predictions_df['Upper Bound'], color='gray', alpha=0.2, label='Prediction Std Dev')
    plt.axhline(y=buy_price_usdt, color='black', linestyle='--', label='Buy Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Cryptocurrency Price Prediction')
    plt.legend()
    plt.show()

    # Calculate time to profit
    time_to_profit = calculate_time_to_profit(investment_time, profitable_time)

    return predictions_df, std_dev, profitable_time, time_to_profit