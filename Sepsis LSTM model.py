#import packages
import pandas as pd
import numpy as np

#to plot within notebook
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#read the file
df = pd.read_csv(r'C:\Users\Hp\Desktop\Sepsis100.csv')
#print the file
print(df)

#importing required libraries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

#creating dataframe
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['index', 'SepsisLabel'])
for i in range(0,len(data)):
    new_data['index'][i] = data['index'][i]
    new_data['SepsisLabel'][i] = data['SepsisLabel'][i]

#setting index
new_data.index = new_data.index
new_data.drop('index', axis=1, inplace=True)

#creating train and test sets
dataset = new_data.values

train = dataset[0:28000,:]
valid = dataset[28000:,:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

model.fit(x_train, y_train, epochs=2, batch_size=250,shuffle=True)

inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
Sepsis_label = model.predict(X_test)
Sepsis_label = scaler.inverse_transform(Sepsis_label)


rms=np.sqrt(np.mean(np.power((valid-Sepsis_label),2)))
print(rms)

#for plotting
train = new_data[27500:28000]
valid = new_data[28000:]
valid['Predictions'] = Sepsis_label
plt.figure(figsize=(50,20))
plt.plot(train['SepsisLabel'])
plt.plot(valid[['SepsisLabel','Predictions']])


model.save('lstm_Sepsis.h5')


r = valid[['SepsisLabel','Predictions']]

r.to_csv(r'C:\Users\Hp\Desktop\Sepsispred.csv')





