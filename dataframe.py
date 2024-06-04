mport pandas as pd

data = {'Name': ['John', 'Emily', 'Michael', 'Sophia', 'Daniel'],
        'Marks': [85, 78, 92, 88, 95],
        'CGPA': [8.5, 8.0, 9.2, 8.8, 9.5],
        'Percentage': [85, 78, 92, 88, 95]}

df = pd.DataFrame(data)

from sklearn.linear_model import LinearRegression

X = df[['Marks', 'CGPA']]  # Features
y = df['Percentage']  # Target variable

model = LinearRegression()
model.fit(X, y)

from sklearn.metrics import mean_squared_error

y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
print("Mean Squared Error:", mse)