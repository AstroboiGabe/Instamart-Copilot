import sqlite3
import pandas as pd

conn = sqlite3.connect('data/instamart.db')

with open('sql/queries.sql','r') as file:
    script = file.read()

# Split on semicolons so each query runs separately
statements = [s.strip() for s in script.split(';') if s.strip()]

results = {}

for i, stmt in enumerate(statements, start=1):
    df = pd.read_sql_query(stmt, conn)
    results[f"Q{i}"] = df
    print(f"\n--- Q{i}: {len(df)} rows ---")
    print(df.head(10))

    if i==2:
        print("\nPivot table representation of 2nd query\n")
        pivot=pd.pivot_table(
            df,
            values='revenue',
            index='city',
            columns='category',
            aggfunc=['sum','mean','count'],
            margins=True
        )
        print(pivot)
    
conn.close()