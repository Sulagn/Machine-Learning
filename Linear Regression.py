import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Score' : [12, 25, 32, 40, 50, 55, 65, 72, 80, 90]
}
df = pd.DataFrame(data)

X = df[['Hours_Studied']]
y = df['Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Predictions:", y_pred)

mse= mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

X_1d = X['Hours_Studied'].values
y_pred_full = model.predict(X).flatten()

plt.scatter(X_1d, y, color='blue', label='Actual Scores')
plt.plot(X_1d, y_pred_full, color='red', label='Predicted Line')
plt.xlabel('Hours_Studied')
plt.ylabel('Score')
plt.title('Hours Studied vs Score Prediction')
plt.legend()
plt.show()