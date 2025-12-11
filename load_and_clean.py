import pandas as pd
import sqlite3

df = pd.read_csv("Churn.csv")
print("Loaded raw dataset.")
print("Initial shape:", df.shape)
print(df.head(), "\n")

df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(r"\s+", "_", regex=True)
print("Cleaned Column Names:", list(df.columns), "\n")

numeric_cols = [
    "Call_Failure", "Complains", "Subscription_Length", "Charge_Amount",
    "Seconds_of_Use", "Frequency_of_use", "Frequency_of_SMS",
    "Distinct_Called_Numbers", "Age", "Customer_Value"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median())

df["Churn"] = df["Churn"].astype(int)

print("Missing values after cleaning:")
print(df.isna().sum(), "\n")

df.to_csv("cleaned_churn.csv", index=False)
print("Saved cleaned_churn.csv in project folder.\n")

df["customerID"] = range(1, len(df) + 1)

customers = df[["customerID", "Age", "Age_Group", "Status"]]

services = df[[
    "customerID", "Call_Failure", "Complains", "Seconds_of_Use",
    "Frequency_of_use", "Frequency_of_SMS", "Distinct_Called_Numbers",
    "Tariff_Plan"
]]

contracts = df[[
    "customerID", "Subscription_Length", "Charge_Amount", "Customer_Value"
]]

churn_labels = df[["customerID", "Churn"]]

conn = sqlite3.connect("churn.db")

customers.to_sql("customers", conn, if_exists="replace", index=False)
services.to_sql("services", conn, if_exists="replace", index=False)
contracts.to_sql("contracts", conn, if_exists="replace", index=False)
churn_labels.to_sql("churn_labels", conn, if_exists="replace", index=False)

print("Created churn.db with normalized tables.\n")


print("=== SQL Insight #1: Churn by Tariff Plan ===")
query1 = """
SELECT s.Tariff_Plan,
       COUNT(*) AS total_customers,
       SUM(cl.Churn) AS churned,
       ROUND(1.0 * SUM(cl.Churn)/COUNT(*), 3) AS churn_rate
FROM services s
JOIN churn_labels cl USING(customerID)
GROUP BY s.Tariff_Plan;
"""
print(pd.read_sql_query(query1, conn), "\n")


print("=== SQL Insight #2: Avg Charge & Subscription Length by Churn ===")
query2 = """
SELECT cl.Churn,
       ROUND(AVG(c.Charge_Amount), 2) AS avg_charge,
       ROUND(AVG(c.Subscription_Length), 2) AS avg_length
FROM contracts c
JOIN churn_labels cl USING(customerID)
GROUP BY cl.Churn;
"""
print(pd.read_sql_query(query2, conn), "\n")

conn.close()
print("Database connection closed.")
print("=== CLEANING + DATABASE BUILD COMPLETE ===")
