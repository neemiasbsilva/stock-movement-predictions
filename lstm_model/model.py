from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

def get_model(X_train):
    model = Sequential()

    model.add(LSTM(units=200, return_sequences=True,
        input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.5))

    model.add(LSTM(units=150, return_sequences=True))
    model.add(Dropout(0.25))

    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.25))

    model.add(LSTM(units=20))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))

    return model