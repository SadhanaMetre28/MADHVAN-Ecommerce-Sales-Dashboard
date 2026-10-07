# MADHVAN E-Commerce Sales Analytics

## Project Overview
An end-to-end data analytics project that studies e-commerce sales, profit, quantity, category, regional and payment-mode performance.

### Tools
- Power BI — interactive dashboard
- Python — analysis and reproducibility
- Pandas / NumPy — data cleaning and transformation
- Matplotlib — charts
- Scikit-learn — basic customer segmentation

## Dataset
- `Orders.csv`: 500 order records
- `Details.csv`: 1,500 transaction/detail records
- Period: January–December 2018
- Join key: `Order ID`

## Key Results
- Total Sales: ₹437,771
- Total Profit: ₹36,963
- Total Quantity: 5,615
- Overall Profit Margin: 8.44%
- Highest-sales state: Maharashtra
- Highest-sales category: Clothing
- Highest-sales month: January 2018

## Dashboard
The Power BI dashboard contains KPI cards and visualizations for:
- Sales
- Profit
- Quantity
- Sales by state
- Quantity by category
- Monthly profit
- Sales/customer analysis
- Payment modes
- Profit by category

## Python Code
`ecommerce_sales_analysis.py`:
1. Loads and cleans the datasets.
2. Joins Orders and Details using Order ID.
3. Creates month, quarter and profit-margin fields.
4. Creates monthly, state and category summaries.
5. Generates charts.
6. Performs basic K-Means customer segmentation using Recency, Frequency and Monetary features.

## Run the Project
Install Python 3.10+ and run:

```bash
pip install -r requirements.txt
python ecommerce_sales_analysis.py
```

Keep `Orders.csv`, `Details.csv`, `ecommerce_sales_analysis.py` and `requirements.txt` in the same folder.

The script creates an `output` folder containing summary CSV files, charts and customer segments.

## Limitations
The data covers one historical year, so findings should not be treated as current business performance. The K-Means segmentation is an exploratory ML component and should be validated with larger and newer datasets before business decisions.

## Author
MADHVAN E-Commerce Sales Analytics Project
