import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# print(os.getcwd())
df_insurance = pd.read_csv("5_logistic_regression/insurance_data.csv")
# print(df_insurance)

plt.scatter(df_insurance["age"], df_insurance["bought_insurance"])
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_insurance[["age"]], df_insurance.bought_insurance, train_size=0.9)
print(X_test)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_train, y_train)

print(model.predict(X_test))
print(model.score(X_test, y_test))

from sklearn import datasets

data_set = datasets.load_breast_cancer()
print(data_set.feature_names)