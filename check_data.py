from data import execute_query

def check_transaction_data():
    # Check the first 10 rows
    query = '''
    SELECT 
        "TransactionID",
        "Product",
        "Price",
        "Quantity",
        ("Price" * "Quantity") as total_amount
    FROM "sales_transactions"
    ORDER BY total_amount DESC
    LIMIT 10
    '''
    
    results = execute_query(query)
    print("Top 10 transactions by total amount:")
    print("-" * 80)
    for row in results:
        print(f"ID: {row['TransactionID']}")
        print(f"Product: {row['Product']}")
        print(f"Price: {row['Price']}")
        print(f"Quantity: {row['Quantity']}")
        print(f"Total: {row['total_amount']}")
        print("-" * 80)

if __name__ == "__main__":
    check_transaction_data()
