import pandas as pd

df = pd.read_csv("heart_disease_uci.csv")

print("Ukuran dataset:", df.shape)
print("\nNama kolom:")
print(df.columns.tolist())

print("\nLima baris pertama:")
print(df.head())

print("\nJumlah nilai kosong:")
print(df.isna().sum())

print("\nTipe data:")
print(df.dtypes)
