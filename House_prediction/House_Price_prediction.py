import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\itsme\Desktop\large_house_prediction.csv")

X = df[["Area_sqft", "Bedrooms", "Age_years"]]
y = df["Price"]

model = LinearRegression()
model.fit(X, y)

Area_sqft = float(input("Enter The Area Sqft :"))
Bedrooms = int(input("Enter The Bedrooms :"))
Age_years = float(input("Enter The Age Years :"))

data = pd.DataFrame({"Area_sqft": [Area_sqft], "Bedrooms": [Bedrooms], "Age_years": [Age_years]})

Price = model.predict(data)
print(f"Price of a House with {Bedrooms} Bedrooms, Area {Area_sqft} and {Age_years} years Age is {int(Price[0])}")
