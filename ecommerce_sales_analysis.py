"""
MADHVAN E-COMMERCE SALES ANALYTICS
Input: Orders.csv + Details.csv
Output: summary CSV files, charts and customer segments.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
OUT = BASE / "output"
CHARTS = OUT / "charts"
OUT.mkdir(exist_ok=True)
CHARTS.mkdir(exist_ok=True)

def load_data():
    orders = pd.read_csv(BASE / "Orders.csv")
    details = pd.read_csv(BASE / "Details.csv")
    orders["Order Date"] = pd.to_datetime(
        orders["Order Date"], dayfirst=True, errors="coerce"
    )
    for df in (orders, details):
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str).str.strip()
    df = details.merge(orders, on="Order ID", how="left", validate="many_to_one")
    if df["Order Date"].isna().any():
        raise ValueError("Some Order IDs could not be matched.")
    df["Month"] = df["Order Date"].dt.to_period("M").astype(str)
    df["Quarter"] = "Q" + df["Order Date"].dt.quarter.astype(str)
    df["Profit Margin %"] = np.where(
        df["Amount"] != 0, df["Profit"] / df["Amount"] * 100, 0
    )
    return df

def analyze(df):
    monthly = df.groupby("Month", as_index=False).agg(
        Sales=("Amount", "sum"), Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum")
    )
    state = df.groupby("State", as_index=False).agg(
        Sales=("Amount", "sum"), Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum")
    ).sort_values("Sales", ascending=False)
    category = df.groupby(["Category", "Sub-Category"], as_index=False).agg(
        Sales=("Amount", "sum"), Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum")
    ).sort_values("Sales", ascending=False)

    monthly.to_csv(OUT / "monthly_sales_profit.csv", index=False)
    state.to_csv(OUT / "state_sales.csv", index=False)
    category.to_csv(OUT / "category_analysis.csv", index=False)

    # Monthly sales chart
    plt.figure(figsize=(9, 5))
    plt.plot(monthly["Month"], monthly["Sales"], marker="o")
    plt.title("Monthly Sales Trend")
    plt.xlabel("Month"); plt.ylabel("Sales Amount")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(CHARTS / "monthly_sales_trend.png", dpi=160)
    plt.close()

    # Top states chart
    top = state.head(10).sort_values("Sales")
    plt.figure(figsize=(9, 5))
    plt.barh(top["State"], top["Sales"])
    plt.title("Top 10 States by Sales")
    plt.xlabel("Sales Amount")
    plt.tight_layout()
    plt.savefig(CHARTS / "top_states_sales.png", dpi=160)
    plt.close()

    # Category chart
    cat = df.groupby("Category", as_index=False)["Amount"].sum().sort_values("Amount")
    plt.figure(figsize=(8, 5))
    plt.barh(cat["Category"], cat["Amount"])
    plt.title("Sales by Category")
    plt.xlabel("Sales Amount")
    plt.tight_layout()
    plt.savefig(CHARTS / "sales_by_category.png", dpi=160)
    plt.close()

    # Payment-mode chart
    pay = df.groupby("PaymentMode", as_index=False)["Amount"].sum().sort_values("Amount")
    plt.figure(figsize=(8, 5))
    plt.bar(pay["PaymentMode"], pay["Amount"])
    plt.title("Sales by Payment Mode")
    plt.xlabel("Payment Mode"); plt.ylabel("Sales Amount")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(CHARTS / "sales_by_payment_mode.png", dpi=160)
    plt.close()

    return monthly, state, category

def customer_segmentation(df):
    # Basic ML: RFM-style customer segmentation with K-Means.
    customer = df.groupby("CustomerName", as_index=False).agg(
        LastOrder=("Order Date", "max"),
        Frequency=("Order ID", "nunique"),
        Monetary=("Amount", "sum")
    )
    customer["Recency"] = (df["Order Date"].max() - customer["LastOrder"]).dt.days
    X = customer[["Recency", "Frequency", "Monetary"]]
    X = StandardScaler().fit_transform(X)
    model = KMeans(n_clusters=4, random_state=42, n_init=10)
    customer["Segment"] = model.fit_predict(X) + 1
    customer.to_csv(OUT / "customer_segments.csv", index=False)
    return customer

def main():
    df = load_data()
    monthly, state, category = analyze(df)
    customer_segmentation(df)

    sales = df["Amount"].sum()
    profit = df["Profit"].sum()
    qty = df["Quantity"].sum()
    margin = profit / sales * 100
    print("=== MADHVAN E-COMMERCE SALES ANALYTICS ===")
    print(f"Total Sales: {sales:,.0f}")
    print(f"Total Profit: {profit:,.0f}")
    print(f"Total Quantity: {qty:,.0f}")
    print(f"Profit Margin: {margin:.2f}%")
    print(f"Top State: {state.iloc[0]['State']}")
    print(f"Top Category: {df.groupby('Category')['Amount'].sum().idxmax()}")
    print(f"Best Sales Month: {monthly.loc[monthly['Sales'].idxmax(), 'Month']}")
    print(f"Output folder: {OUT}")

if __name__ == "__main__":
    main()
