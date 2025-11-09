import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder


df = pd.read_csv(r"C:\Users\itsme\Desktop\credit_fraud_no_pca_1k.csv")
# print(df.head())

card_encoder = LabelEncoder()
merchant_encoder = LabelEncoder()
location_encoder = LabelEncoder()
device_encoder = LabelEncoder()

scaler = StandardScaler()

df["cardType_encoded"] = card_encoder.fit_transform(df["CardType"])
df["merchant_encoded"] = merchant_encoder.fit_transform(df["Merchant"])
df["location_encoded"] = location_encoder.fit_transform(df["Location"])
df["device_encoded"] = device_encoder.fit_transform(df["DeviceType"])


pca = PCA(n_components=3)
X_pca = pca.fit_transform(df[["cardType_encoded", "merchant_encoded", "location_encoded", "device_encoded", "IsInternational"]])
# print(df.head(10))
# print(pca.explained_variance_ratio_)
# print(X_pca)

df[["PCA_1", "PCA_2", "PCA_3"]] = pd.DataFrame(X_pca) # Very Important
print(df.head())

X = df[["Amount", "Time", "PCA_1", "PCA_2", "PCA_3"]]
y = df["IsFraud"]

model = LogisticRegression()
model.fit(X,y)

Amount = float(input("Enter The Amount:"))
Time = float(input("Enter The Time:"))
CardType = input("Enter either Credit or Debit :")
Merchant = input("Enter Paytm, Swiggy, Myntra, Flipkart, Amazon:")
Location = input("Enter Mumbai, Bangalore, Chennai, Kolkata, Delhi:")
DeviceType = input("Enter iOS, Android, Web:")
IsInternational = int(input("Enter 0 for national 1 for International Payment:"))

card_val = card_encoder.transform([CardType])[0]
merchant_val = merchant_encoder.transform([Merchant])[0]
location_val = location_encoder.transform([Location])[0]
device_val = device_encoder.transform([DeviceType])[0]

# Apply PCA to encoded values
pca_input = [[card_val, merchant_val, location_val, device_val, IsInternational]]
final_pca = pca.transform(pca_input)[0]

# Create dataframe for prediction
df_input = pd.DataFrame([{
    "Amount": Amount,
    "Time": Time,
    "PCA_1": final_pca[0],
    "PCA_2": final_pca[1],
    "PCA_3": final_pca[2]
}])

# Predict fraud
result = model.predict(df_input)[0]
print("Fraud" if result == 1 else "Not Fraud")