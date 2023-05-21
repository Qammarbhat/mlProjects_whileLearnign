# Learn to implement the model $f_{w,b}$ for linear regression with one variable

import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("resources/deeplearning.mplstyle")
print(os.getcwd())

# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

# Number of training examples:  m
# m is the number of training examples
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"number of training examples is {m}")

# One can also use the Python len() function as shown below.
# m is the number of training examples
m = len(x_train)
print(f"Number of training examples is: {m}, using len() function here")


# Training example x_i, y_i
i = 0
x_i = x_train[i]
y_i = y_train[i]
print(f"x^{i}, y^{i} = {x_i}, {y_i}"  )
print("====================================")
# plt.ion()
# Plotting the data
plt.scatter(x_train, y_train, marker = "x", c ="r")

# tittle
plt.title("Housing Prices")

# X Label
plt.xlabel("Size of house(1000 sqft)")

# y Label
plt.ylabel("House price (1000s of dollars)")

plt.show()

# $$ f_{w,b}(x^{(i)}) = wx^{(i)} + b \tag{1}$$, using this formula to compute model

# w = 100 
# b = 100 
# the model predicted wrong output so after tuning the w and b variable sometime we gwt

w = 200
b = 100

def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      y (ndarray (m,)): target values
    """

    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb

# Calling compute model
tmp_f_wb = compute_model_output(x_train, w, b)

# Plot model prediction
plt.plot(x_train, tmp_f_wb, c="b", label="Our Prediction")

# Plot the data points
plt.scatter(x_train, y_train, marker = "x" , c = "r", label = "Actual Values")

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()


#   Let's predict the price of a house with 1200 sqft. (i.e 1.2)
x_i = 1.2
cost_1200sqft = w * x_i +b
print(f"cost of 1200 sqft is : {cost_1200sqft} thousand Dollars")










