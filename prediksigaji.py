import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
data = {'Year of Experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Skill': [5, 6, 7, 8, 9, 10, 8, 9, 10, 8],
        'Performance Value': [7, 8, 9, 10, 8, 9, 10, 8, 9, 10],
        'Salary Increase': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
df = pd.DataFrame(data)

X = df[['Year of Experience', 'Skill', 'Performance Value']]
y = df['Salary Increase']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X = df[['Year of Experience', 'Skill', 'Performance Value']]
y = df['Salary Increase']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

train_score = regressor.score(X_train, y_train)
test_score = regressor.score(X_test, y_test)
print("Train Score: ", train_score)
print("Test Score: ", test_score)

employee_data = [[5, 9, 9]]
prediction = regressor.predict(employee_data)
print("Predicted Salary Increase: ", prediction[0])
