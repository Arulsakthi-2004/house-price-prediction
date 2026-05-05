
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd

url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

df.head()
print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


df = df.fillna(df.mean(numeric_only=True))


df = pd.get_dummies(df, drop_first=True)

X = df.drop("medv", axis=1)
y = df["medv"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)



model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

comparison = pd.DataFrame({
    "Actual": y_test[:10].values,
    "Predicted": y_pred[:10]
})


comparison.plot(kind="bar")
plt.title("Actual vs Predicted Prices (Sample)")
plt.xlabel("Sample Index")
plt.ylabel("House Price")
plt.show()