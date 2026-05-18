import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import Input


def create_dataset(dataset, look_back):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i+look_back, 0])
        Y.append(dataset[i+look_back, 0])
    return np.array(X), np.array(Y)


def train_and_predict(data, look_back=5, epochs=10):

    # =========================
    # SCALING
    # =========================
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # =========================
    # TRAIN-TEST SPLIT
    # =========================
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    X_train, y_train = create_dataset(train_data, look_back)
    X_test, y_test = create_dataset(test_data, look_back)

    # ✅ Safety check
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Dataset too small. Increase data or reduce look_back.")

    # Reshape for LSTM
    X_train = X_train.reshape(X_train.shape[0], look_back, 1)
    X_test = X_test.reshape(X_test.shape[0], look_back, 1)

    # =========================
    # MODEL
    # =========================
    model = Sequential()
    model.add(Input(shape=(look_back, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    # =========================
    # TRAIN
    # =========================
    model.fit(X_train, y_train, epochs=epochs, batch_size=16, verbose=0)

    # =========================
    # EVALUATION
    # =========================
    train_pred = model.predict(X_train, verbose=0)
    test_pred = model.predict(X_test, verbose=0)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_mae = mean_absolute_error(y_test, test_pred)

    # =========================
    # FUTURE PREDICTION
    # =========================
    future_input = scaled_data[-look_back:]
    future_input = future_input.reshape(1, look_back, 1)

    future_predictions = []

    for _ in range(12):
        next_pred = model.predict(future_input, verbose=0)
        future_predictions.append(next_pred[0][0])

        next_pred = next_pred.reshape(1, 1, 1)
        future_input = np.append(future_input[:, 1:, :], next_pred, axis=1)

    future_predictions = scaler.inverse_transform(
        np.array(future_predictions).reshape(-1, 1)
    )

    return future_predictions, train_rmse, test_rmse, test_mae