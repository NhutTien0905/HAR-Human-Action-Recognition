import os
import pickle
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense,Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split

# Đọc dữ liệu
no_of_timesteps = 10
lst_data = os.listdir("data2")
X = []
y = []
print(lst_data)
# ['CLAP.txt', 'HIT.txt', 'JUMP.txt', 'RUN.txt', 'SIT.txt', 'STAND.txt', 'THROW_HAND.txt', 'WALK.txt', 'WAVE_HAND.txt']

if "data.pickle" in lst_data:
    print("Loading data from data.pickle")
    with open("data2/data.pickle", "rb") as f:
        X, y = pickle.load(f)
else:
    print("Creating data.pickle")
    for i, path in enumerate(lst_data):
        df = pd.read_csv("data2/" + path)
        dataset = df.iloc[:,1:].values
        n_sample = len(dataset)
        for j in range(no_of_timesteps, n_sample):
            X.append(dataset[j-no_of_timesteps:j,:])
            y.append(i)

    X, y = np.array(X), np.array(y)
    with open("data/data.pickle", "wb") as f:
        pickle.dump((X, y), f)

print(X.shape, y.shape)

# create one hot encoding
num_classes = len(lst_data)
y = np.eye(num_classes)[y]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create model
model  = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = num_classes, activation="softmax"))
model.compile(optimizer="adam", metrics = ['accuracy'], loss = "binary_crossentropy")

model.fit(X_train, y_train, epochs=20, batch_size=32,validation_data=(X_test, y_test))
model.save("model/model_1.h5")