import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(r"C:\Users\itsme\Desktop\iris_petal_dataset.csv")

encoder = LabelEncoder()
df["species_encoder"] = encoder.fit_transform(df["species"])

# print(df.head(102))

X = df[["petal length (cm)", "petal width (cm)"]]
y = df["species_encoder"]

model = KNeighborsClassifier(n_neighbors = 3)
model.fit(X, y)

lenght = float(input("Enter The Lenght :"))
width = float(input("Enter The Width :"))

df_in = pd.DataFrame({"petal length (cm)": [lenght], "petal width (cm)": [width]})

result = model.predict(df_in)

print("Setosa" if result == 0 else( "Versicolor" if result == 1 else "Virginica") )