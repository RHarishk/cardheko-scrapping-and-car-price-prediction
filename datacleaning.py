import pandas as pd
import re
import numpy as np

ah=pd.read_csv("/content/cardekho_data_ahmedabad.csv")
aj=pd.read_csv("/content/cardekho_data_ajmerSpecs.csv")
ba=pd.read_csv("/content/cardekho_data_bangalore.csv")
ch=pd.read_csv("/content/cardekho_data_chennai.csv")
co=pd.read_csv("/content/cardekho_data_coimbatore.csv")
hy=pd.read_csv("/content/cardekho_data_hyderabad.csv")
ko=pd.read_csv("/content/cardekho_data_kolkata.csv")
pu=pd.read_csv("/content/cardekho_data_pune.csv")
ja=pd.read_csv("/content/cardekho_data_jaipur.csv")
cg=pd.read_csv("/content/cardekho_data_chandigarh.csv")

df = pd.concat([ah,aj,ba,ch,co,hy,ko,pu,cg,ja], ignore_index=True)

df

df["make_year"]=df["Used_Car_Name"].str.slice(0,5)
df["Used_Car_Name"] = df["Used_Car_Name"].str.slice(4)

df["Used_Car_Name"]=df["Used_Car_Name"].str.lstrip()

df[["Brand", "Model"]] = df["Used_Car_Name"].str.split(" ", n=1, expand=True)

df = df[df["Overview"] != "Not Available"]

df.duplicated().sum()

df=df.drop_duplicates()

df.drop("Used_Car_Name",axis=1,inplace=True)

def convert_price(price_str):
    """Convert price string to numeric value based on Lakhs and Crores."""
    match = re.search(r'([\d\.]+)\s*(Lakh|Crore)?', price_str)
    if match:
        value, unit = match.groups()
        value = float(value)
        if unit == "Lakh":
            return value * 100000
        elif unit == "Crore":
            return value * 10000000
        else:
            return value *1000  # Already in base unit
    return None

# Apply conversion function to the Price column
df["price"] = df["Price"].apply(convert_price)


df[['km_driven', 'fuel_type', 'transmission']] = df['KM_Driven'].str.split(' • ', expand=True)
df["km_driven"]=df["km_driven"].str.slice(0,-5)
df["km_driven"]=df["km_driven"].str.replace(",","",regex=True)

df["Model"]=df["Model"].str.extract(r'^([\S]+) ')
df["registration_year"] = df["Overview"].str.extract(r'Registration Year\n(?:\w+\s)?(\d{4})')
df["insurance_type"] = df["Overview"].str.extract(r'Insurance\n([\w\s]+)')
df["no_of_seats"]=df["Overview"].str.extract(r'Seats\n(\d+) ')
df["Ownership"]=df["Overview"].str.extract(r'Ownership\n([\w\s]+),')
df["power"] = df["Specification"].str.extract(r'Power\n([\d\.]+) bhp')
df["cc"] = df["Specification"].str.extract(r'Engine\n([\d\.]+) cc')
df["Mileage"] = df["Specification"].str.extract(r'Mileage\n([\d\.]+) kmpl')
df["top_speed"] = df["Specification"].str.extract(r'Top Speed\n([\d\.]+) kmph')

df.drop("Price",axis=1,inplace=True)
df.drop("Overview",axis=1,inplace=True)
df.drop("Specification",axis=1,inplace=True)
df.drop("KM_Driven",axis=1,inplace=True)
df.drop("top_speed",axis=1,inplace=True)

df

df.info()

df["cc"].fillna(0,inplace=True)

df.dropna(subset=["Ownership"], inplace=True)

df["make_year"]=df["make_year"].astype(int)
df["cc"] = df["cc"].astype(int)
df["km_driven"] = df["km_driven"].astype(int)
df["price"] = df["price"].astype(float)
df["Mileage"]=df["Mileage"].astype(float)

df.loc[(df["Mileage"].isna()) & (df["cc"] >= 1500), "Mileage"] = 14
df.loc[(df["Mileage"].isna()) & (df["cc"] <1500) & (df["cc"] >1000), "Mileage"] = 18
df.loc[(df["Mileage"].isna()) & (df["cc"] <=1000), "Mileage"] = 20

df["no_of_seats"].fillna("5",inplace=True)

df["power"].fillna(df["cc"]*0.08,inplace=True)

df['registration_year'].fillna(df['make_year'],inplace=True)

df.info()

df["registration_year"]=df["registration_year"].astype(int)
df["price"]=df["price"].astype(int)
df["power"]=df["power"].astype(float)

df.columns=df.columns.str.lower()
