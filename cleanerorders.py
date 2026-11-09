import numpy as np
import pandas as pd
import sqlite3

conn=sqlite3.connect(':memory:')
print("Connection created: ",conn)

df=pd.read_csv('data/raw/orders.csv')
print(df.head())
print(df.info())
print("-"*50)
print(df.shape)

dfp=pd.read_csv('data/raw/products.csv')
print(dfp.head())

df.to_sql('orders',conn,index=False,if_exists='replace')
dfp.to_sql('products',conn,index=False,if_exists='replace')

def sql(query):
    return pd.read_sql_query(query,conn)

df["order_ts"]=pd.to_datetime(df["order_ts"])
print(df["order_ts"].dtype)

df=df.drop_duplicates()
print(df.info())

df=df.sort_values("order_ts").drop_duplicates(subset=["order_id"],keep="last")

query="""
SELECT o.quantity, o.discount_pct, p.price_inr FROM orders o
INNER JOIN products p ON o.product_id = p.product_id 
"""
calctab=sql(query)

calctab["calculation"]=calctab["price_inr"]*calctab["quantity"]*(1-(calctab["discount_pct"]/100))

df["order_amount"]=pd.to_numeric(df["order_amount"], errors="coerce")
print(df["order_amount"].dtype)

df["order_amount"]=df["order_amount"].fillna(round(calctab["calculation"],2))

df.loc[df["delivery_minutes"]<0, "delivery_minutes"]=np.nan
df["delivery_minutes"]=df["delivery_minutes"].fillna(df["delivery_minutes"].median())

df.groupby('status')["delivery_minutes"].transform("median")

print(df['status'].value_counts())
print("order amount sum value for completed orders is ",df.groupby('status')["order_amount"].sum())

df.to_csv('data/clean/orders.csv',index=False)

#VALIDATION PROCESS
i=3
if df["order_ts"].dtype=="datetime64[us]":
    i+=1
    print("Data type of order_ts column is validated ✅")

if df.shape[0]==12309:
    i+=1
    print("Number of rows verified ✅")

if df["order_amount"].dtype==float:
    i+=1
    print("Order amount data type changed to numeric ✅")

if df["order_amount"].count()==df.shape[0]:
    i+=1
    print("Order amount column has no null values ✅")

statuses=df.groupby("status")["order_id"].count()
if statuses["completed"]==11110:
    if statuses["cancelled"]==1199:
        if df.shape[0]==statuses["completed"]+statuses["cancelled"]:
            i+=1
            print("Status count verified with no pending values in the column ✅")

if (df["delivery_minutes"]<0).sum()==0:
    i+=1
    print("All delivery minutes values are positive ✅")

totalamount=df.groupby('status')["order_amount"].sum()
if totalamount["completed"]==2998668.80:
    i+=1
    print("Total amount for completed orders is verified ✅")

print(f"{i} number of validations has been validated for customers & orders table cleaning\n")
if i==10:
    print("ALL CHECKPOINTS PASSED")

conn=sqlite3.connect('data/instamart.db')

df["order_ts"]=df["order_ts"].astype(str)
df.to_sql('orders',conn,index=False,if_exists='replace')
dfp.to_sql('products',conn,index=False,if_exists='replace')
conn.close()
