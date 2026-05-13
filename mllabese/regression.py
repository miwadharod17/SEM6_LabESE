import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


data = {
    "Hours": [1, 2, 3, 4, np.nan, 6, 7, 8],
    "Gender": ["M", "F", "M", "F", "M", "F", "M", "F"],
    "Marks": [10, 20, 30, 40, 50, 60, 70, 80]
}

df = pd.DataFrame(data)
print("Original Dataset:")
print(df)

#preprocessing

imputer = SimpleImputer(strategy = 'mean')
df["Hours"] = imputer.fit_transform(df[["Hours"]])

encoder = LabelEncoder()
df["Gender"] = encoder.fit_transform(df["Gender"])


#model training

X = df[["Hours","Gender"]]
Y = df["Marks"]

model = LinearRegression()

X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size = 0.2 , random_state = 42)
model.fit(X_train,Y_train)
predictions = model.predict(X_test)

print("\n Predictions")
print(predictions)

#metrics

mae = mean_absolute_error(Y_test , predictions)
mse = mean_squared_error(Y_test , predictions)
r2 = r2_score(Y_test , predictions)

print("\nMAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)

#pyplot

plt.scatter(df["Hours"],df["Marks"] ,label = "Actual Data")
plt.plot(df["Hours"], model.predict(X), marker='o')

plt.xlabel("Study Hours")
plt.ylabel("Marks")

plt.title("Linear Regression")

plt.legend()

plt.show()

