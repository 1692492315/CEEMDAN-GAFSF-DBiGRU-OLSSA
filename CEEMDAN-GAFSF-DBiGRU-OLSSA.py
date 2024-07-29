import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, GRU, Bidirectional
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import math
from OLSSA import OLSSA


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


def creat_setdata(df, Time_step):
    trainX = df[:, :]
    scalarX = StandardScaler()
    trainX = scalarX.fit_transform(trainX)
    trainY = df[:, 0].reshape(-1, 1)
    scalarY = StandardScaler()
    trainY = scalarY.fit_transform(trainY)
    dataX, dataY = [], []
    for i in range(len(trainY) - Time_step):
        a = trainX[i:(i + Time_step)]
        dataX.append(a)
        dataY.append(trainY[i + Time_step])

    return np.array(dataX), np.array(dataY), scalarY


_init()


Time_step = 6
predicted_values, real_values = np.zeros([300-Time_step, 1]), np.zeros([300-Time_step, 1])
for n in range(6):
    def fitness_function(solution):
        training_set1 = get_value('training_set')
        x_train1, y_train1, scalarY2 = creat_setdata(training_set1, Time_step)
        x_train1 = np.reshape(x_train1, (x_train1.shape[0], x_train1.shape[1], x_train1.shape[2]))

        model1 = Sequential()
        model1.add(Bidirectional(GRU(int(solution[0]), return_sequences=True, input_shape=(x_train1.shape[1], x_train1.shape[2]))))
        model1.add(Dropout(0.05))
        model1.add(Bidirectional(GRU(2 * int(solution[0]))))
        model1.add(Dropout(0.05))
        model1.add(Dense(1))
        model1.compile(optimizer=tf.keras.optimizers.Adam(float(solution[1])), loss='mean_squared_error')
        model1.fit(x_train1, y_train1, batch_size=32, epochs=100, verbose=0)

        prediction1 = model1.predict(x_train1)
        train_predicted = scalarY2.inverse_transform(prediction1.reshape(-1, 1))
        train_real = scalarY2.inverse_transform(y_train1.reshape(-1, 1))
        fitness_mse = mean_squared_error(train_predicted, train_real)
        return fitness_mse


    # Read data and divide datasets
    dataset = pd.read_excel(io="Data_after_feature_selection_and_decomposition.xlsx", sheet_name=n)
    training_set = dataset.values[:900, :]
    test_set = dataset.values[900:, :]
    set_value('training_set', training_set)

    # Construct input-output data structures
    x_train, y_train, scalarY1 = creat_setdata(training_set, Time_step)
    x_test, y_test, scalarY = creat_setdata(test_set, Time_step)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    # Set the parameters of the optimization algorithm
    pop = 15
    MaxIter = 20
    dim = 2
    lb = [6, 0.001]
    ub = [18, 0.01]
    fobj = fitness_function

    # Prediction model parameters optimization
    GbestScore, GbestPositon, Curve = OLSSA(pop, dim, lb, ub, MaxIter, fobj)
    GbestPositon = GbestPositon.reshape(2, 1)
    print('optimal fitness value：', GbestScore)
    print('optimal solution：', GbestPositon)

    # Training and forecasting
    model = Sequential()
    model.add(Bidirectional(GRU(int(GbestPositon[0]), return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2]))))
    model.add(Dropout(0.05))
    model.add(Bidirectional(GRU(2 * int(GbestPositon[0]))))
    model.add(Dropout(0.05))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(float(GbestPositon[1])), loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=0)
    prediction = model.predict(x_test)
    subsequence_predicted = scalarY.inverse_transform(prediction.reshape(-1, 1))
    subsequence_real = scalarY.inverse_transform(y_test.reshape(-1, 1))
    predicted_values = predicted_values + subsequence_predicted
    real_values = real_values + subsequence_real


# Plot the comparison curves of real and predicted data
plt.plot(real_values, color='red', label='Real wind speed')
plt.plot(predicted_values, color='blue', label='Predicted wind speed')
plt.title('Wind speed prediction')
plt.xlabel('Time (10 min)')
plt.ylabel('Wind Speed (m/s)')
plt.legend()
plt.show()


# Evaluation metrics
rmse = math.sqrt(mean_squared_error(predicted_values, real_values))
mae = mean_absolute_error(predicted_values, real_values)
mape = np.mean(np.abs((predicted_values - real_values)/real_values))
R2 = r2_score(real_values, predicted_values)
print('Root mean squared error: %.4f' % rmse)
print('Mean absolute error: %.4f' % mae)
print('Mean absolute percentage error: %.6f' % mape)
print('Coefficient of determination: %.4f' % R2)