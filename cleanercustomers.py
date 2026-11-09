import numpy as np
import pandas as pd
import sqlite3

df=pd.read_csv('data/raw/customers.csv')
print(df.head())
print(df.info())
print("-"*50)
print(df.shape)

df["signup_date"]=pd.to_datetime(df["signup_date"])
print(df["signup_date"].dtype)

(df.loc[df["age"]>100,["age"]])=np.nan
(df.loc[df["age"]<10,["age"]])=np.nan


df["age"]=df["age"].fillna(df.groupby("city")["age"].transform("median"))

print(df["age"].describe())

df["city"]=df["city"].str.lower().str.strip()
print(df["city"].value_counts())

df["gender"]=df["gender"].str.lower()

age_map={
    "m":"male",
    "f":"female"
}

df["gender"]=df["gender"].replace(age_map)
print(df["gender"].value_counts())
print(df.isnull().sum())

#VALIDATION PROCESS
i=0
cities=df['city'].unique()
if len(cities)==6:
    i+=1
    print("City count validated ✅")

genders=df.groupby("gender")["customer_id"].size()
print(genders)
if genders["female"]==610:
    if genders["male"]==560:
        if df.shape[0]-genders["female"]-genders["male"]==30:
            i+=1
            print("Gender count validated ✅")

if df["age"].min()==18:
    if df["age"].max()==60:
        if df["age"].count()==1200:
            i+=1
            print("Age boundaries verified and has no null values in it ✅")

print(f"{i} number of validations has been validated for customers table cleaning")

df.to_csv("data/clean/customers.csv",index=False)

conn=sqlite3.connect('data/instamart.db')
df.to_sql('customers',conn,index=False,if_exists="replace")
conn.close()
