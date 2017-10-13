import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

# Reading the file to a Panda DataFrame

df = pd.read_csv('siraj-machine-learning/linear_regression_demo/challenge_dataset.txt', names=['A', 'B'])
x_values = df[['A']]
y_values = df[['B']]

regr = linear_model.LinearRegression()
regr.fit(x_values, y_values)

def guess (x_value):
    y_value = regr.predict(x_value)
    return y_value

print(guess(8))

plt.scatter(x_values, y_values)
plt.plot(x_values, regr.predict(x_values))
plt.show()
