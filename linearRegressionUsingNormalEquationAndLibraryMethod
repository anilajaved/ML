import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def normal_equation(X, y):

    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    # equation (X^T X)^-1 X^T y
    w = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    plt.plot(X, y, "r.", label="Data points")
    plt.xlabel("x")
    plt.ylabel("y")

    X_new = np.array([[0], [6]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]
    y_predict = X_new_b.dot(w)

    # Plot the regression line
    plt.plot(X_new, y_predict, "b-", label="Regression line")
    plt.legend()
    plt.show()

    return w

def linear_reg(X, y):

  model = LinearRegression()

  model.fit(X, y)
  # print("Intercept", model.intercept_)
  # print("Slope (w)", model.coef_)

  X_new = np.array([[0], [6]])
  y_predict = model.predict(X_new)

  return [model.intercept_,model.coef_]


number_of_rooms = np.array([[2], [3], [1], [2], [5]])
house_price = np.array([[2000], [4000], [1000], [2500], [6000]])

# Compute weights using the normal equation
weights = normal_equation(number_of_rooms, house_price)
print(f"Weights through pseudoinverse formula: {weights}")

# Compute weights using the normal equation
reg_weights = linear_reg(number_of_rooms, house_price)
print(f"Weights through linear regression: {reg_weights}")


