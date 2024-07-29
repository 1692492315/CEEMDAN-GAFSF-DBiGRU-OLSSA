import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dropout, Dense, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sko.GA import RCGA


tf.random.set_seed(7)


# Define global variable function
def _init():
    global _global_dict
    _global_dict = {}

def set_value(key, value):
    _global_dict[key] = value

def get_value(key):
    try:
        return _global_dict[key]
    except:
        print('read'+key+'failure\r\n')


def DecToBin(i):
    list = []
    while i:
        list.append(i % 2)
        i = i // 2
    list.reverse()

    return list


_init()


def fitness_function(solution):
    training_set1 = get_value('training_set')
    # Obtain feature set based on binary encoding
    train_list = DecToBin(int(round(solution[0], 0)))
    Binstring = (''.join(map(str, train_list))).zfill(7)
    Binlist = [int(char) for char in Binstring]
    train_x = []
    for i in range(len(Binlist)):
        if Binlist[i] == 1:
            train_x.append(training_set1[:, i])
        else:
            pass
    train_x = np.array(train_x).T
    train_y = training_set1[:, 0].reshape(-1, 1)

    # Construct input-output data structures
    scalarX = StandardScaler()
    standard_train_x = scalarX.fit_transform(train_x)
    scalarY = StandardScaler()
    standard_train_y = scalarY.fit_transform(train_y)
    Time_step = 6
    dataX, dataY = [], []
    for i in range(len(standard_train_y) - Time_step):
        a = standard_train_x[i:(i + Time_step)]
        dataX.append(a)
        dataY.append(standard_train_y[i + Time_step])

    x_train = np.array(dataX)
    y_train = np.array(dataY)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))

    # Training and forecasting of the base forecasting model
    model = Sequential()
    model.add(Bidirectional(LSTM(16, input_shape=(x_train.shape[1], x_train.shape[2]))))
    model.add(Dropout(0.05))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=0)
    prediction = model.predict(x_train)
    train_predicted = scalarY.inverse_transform(prediction.reshape(-1, 1))
    train_real = scalarY.inverse_transform(y_train.reshape(-1, 1))
    fitness_mse = 1-mean_squared_error(train_predicted, train_real)
    return fitness_mse


# Read data and divide datasets
dataset = pd.read_excel(io="Wind_speed_data_on_7_turbines.xlsx", sheet_name="Sheet1")
training_set = dataset.values[:900, :]
set_value('training_set', training_set)

fobj = fitness_function
ga = RCGA(func=fobj, n_dim=1, size_pop=10, max_iter=15, prob_mut=0.2, prob_cros=0.5, lb=[1], ub=[127])
best_x, best_y = ga.run()
print('optimal fitness value：', best_y)
print('optimal solution：', best_x)


best_train_list = DecToBin(int(round(best_x[0], 0)))
best_Binstring = (''.join(map(str, best_train_list))).zfill(7)
print(best_Binstring)
best_Binlist = [int(char) for char in best_Binstring]
print('Optimal binary encoding：', best_Binlist)
train_x1 = []
for i in range(len(best_Binlist)):
    if best_Binlist[i] == 1:
        train_x1.append(training_set[:, i])
    else:
        pass

optimal_feature_set = np.array(train_x1).T
optimal_feature_set = pd.DataFrame(optimal_feature_set)
optimal_feature_set.to_excel('optimal_feature_set.xlsx', index=False)

