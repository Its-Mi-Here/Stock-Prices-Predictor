#Importing basic libraries required
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

#Loading dataset
dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")


training_set = dataset_train.iloc[:,1:2]

# Feature Scaling
sc = MinMaxScaler()         
training_set_scaled = sc.fit_transform(training_set)


# Creating a data structure with timesteps of 60 as in previous values will help predict the 61st value
X_train = []
y_train = []
for i in range(60,len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1], 1))


#Importing Keras Libraries
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

#LSTM block structure  
regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1],1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

#Output layer
regressor.add(Dense(units = 1))

#Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#If you don't want to save the weights
#regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


#For saving the history of weights after training the network
from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True)
train_history = regressor.fit( X_train, y_train, validation_data=(X_test, predicted_stock_price), epochs=100, batch_size=32, callbacks = [checkpointer], verbose=1)


#Getting real stock Price
dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[:,1:2].values
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total)-len(dataset_test) - 60 : ].values
inputs = inputs.reshape(-1,1)

inputs = sc.transform(inputs)
X_test = []
for i in range(60,80):
    X_test.append(inputs[i-60:i,0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


#For loading weights instead of training the entire network again.
regressor.load_weights('weights.hdf5' ) 

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


#Visualization
plt.plot(real_stock_price, color = 'red', label = "Real Stock Price")
plt.plot(predicted_stock_price, color = 'blue', label = "Predicted Stock Price")
plt.title("Google stock prices")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.show()