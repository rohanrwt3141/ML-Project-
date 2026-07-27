import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("/home/rohan/Desktop/Crop_Recommendation/Crop_recommendation.csv")


print(df.select_dtypes("int").columns)

print(df.select_dtypes("object").columns)

df["label"].unique()

encoder = OrdinalEncoder(categories = [['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
       'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
       'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple',
       'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']])
y = encoder.fit_transform(df[["label"]])

pipe = Pipeline(steps = [("Scaler", StandardScaler()), ("model", LogisticRegression())])


X = df[['N', 'P', 'K', "temperature", "humidity", "ph", "rainfall"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

pipe.fit(X_train, y_train)

y_pred_t = pipe.predict(X_test)


for i, j, k in zip(df["label"], y_pred_t, y_test):
    if(j != k):
        print(i, " ", j, " ", k)

acc = accuracy_score(y_test, y_pred_t)

print(acc)

print(classification_report(y_test, y_pred_t))

# y_pred_tr = pipe.predict(X_train)

# print(classification_report(y_train, y_pred_tr))

print(pipe.predict(X_test[0:20]))

N = int(input("Enter the Nitrogen Level : "))
P = int(input("Enter the Phosphorus Level : "))
K = int(input("Enter the Potassium Level : "))
temp = float(input("Enter the Temperature : "))
humidity = float(input("Enter the Humidity : "))
ph = float(input("Enter the pH Level : "))
rainfall = float(input("Enter the Rainfall : "))

df_input = pd.DataFrame({"N": [N], "P": [P], "K": [K], "temperature": [temp], "humidity": [humidity], "ph": [ph], "rainfall": [rainfall]})


output = pipe.predict(df_input)

print(round(output[0]))

def get_crop_name(prediction):
    match prediction:
        case 0:
            return "rice"
        case 1:
            return "maize"
        case 2:
            return "chickpea"
        case 3:
            return "kidneybeans"
        case 4:
            return "pigeonpeas"
        case 5:
            return "mothbeans"
        case 6:
            return "mungbean"
        case 7:
            return "blackgram"
        case 8:
            return "lentil"
        case 9:
            return "pomegranate"
        case 10:
            return "banana"
        case 11:
            return "mango"
        case 12:
            return "grapes"
        case 13:
            return "watermelon"
        case 14:
            return "muskmelon"
        case 15:
            return "apple"
        case 16:
            return "orange"
        case 17:
            return "papaya"
        case 18:
            return "coconut"
        case 19:
            return "cotton"
        case 20:
            return "jute"
        case 21:
            return "coffee"


print(get_crop_name(round(output[0])))