import pandas as pd
import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf


tf.random.set_seed(7)


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


# Read data and divide datasets
dataset = pd.read_excel(io="Wind_speed_data_on_7_turbines.xlsx", sheet_name="Sheet1")
training_set = dataset.values[:900, :]
test_set = dataset.values[900:, :]


# Construct input-output data structures
Time_step = 6
x_train, y_train, scalarY1 = creat_setdata(training_set, Time_step)
x_test, y_test, scalarY = creat_setdata(test_set, Time_step)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
y_train = np.reshape(y_train, (y_train.shape[0], ))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]*x_test.shape[2]))
y_test = np.reshape(y_test, (y_test.shape[0], ))


# Training and forecasting
rf = RandomForestRegressor(n_estimators=1000, random_state=50)
rf.fit(x_train, y_train)
prediction = rf.predict(x_test)
predicted_values = scalarY.inverse_transform(prediction.reshape(-1, 1))
real_values = scalarY.inverse_transform(y_test.reshape(-1, 1))


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

