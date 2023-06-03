import numpy as np
import matplotlib.pylab as plt

# f_wb = w*x + b
# dj_dw = derivative of j wrt w 
# dj_db = derivative of j wrt b
# j = cost: 1/m* (f_wb*x[i] - y)**2

# Training Data
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

# Calculating Cost
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = (w * x[i]) + b
        cost = cost + (f_wb - y[i])**2
    total_cost = 1 / (2*m) *cost
    return total_cost


# Computing gradient
def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w* x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = (f_wb - y[i])
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    dj_dw = dj_dw/m
    dj_db = dj_db/m

    return dj_db, dj_dw

# Computing gradient descent
def gradient_descent(x, y, w_in, b_in, learning_rate, num_iterations, cost_function, gradient_function):
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    cost = 0.0

    for i in range(num_iterations):
        dj_db, dj_dw = gradient_function(x, y, w, b)

        b = b - learning_rate*dj_db
        w = w - learning_rate*dj_dw
        cost = compute_cost(x, y, w, b)
        print(f"Iteration : {i}, w: {w}, b: {b}, cost: {cost}")

        # appending the values in j and p history
        if i < 10000:
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w,b])
        # print(J_history, p_history)
    return w, b, J_history, p_history
        

# Initialize parameters
w_init  = 0
b_init = 0
iteration = 10000
tmp_alpha = 1.0e-2
gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha, iteration, compute_cost, compute_gradient)
# w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, 
#                                                     iteration, compute_cost, compute_gradient)
# print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")

