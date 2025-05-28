import os
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Dict, Any, Optional
import json

# Load environment variables
load_dotenv()

# Configure Google's Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the model
MODEL_NAME = "gemini-1.5-flash"
model = genai.GenerativeModel(MODEL_NAME)

# Database schema information with exact column cases
SCHEMA_INFO = """
Database Schema Details (Note: Column names are case-sensitive):

1. TABLE: customer_info
   Description: Stores information about customers.
   Columns (use exact case as shown):
   - "CustomerID" (bigint, PRIMARY KEY): Unique identifier for each customer
   - "Name" (text): Name of the customer
   - "Gender" (text): Gender of the customer
   - "Age" (bigint): Age of the customer
   - "LoyaltyCard" (text): Loyalty card information
   - "AvgSpending" (double precision): Average spending of the customer
   - "VisitFrequency" (bigint): Frequency of visits by the customer

2. TABLE: employee_performance
   Description: Tracks the performance of employees.
   Columns (use exact case as shown):
   - "EmployeeID" (bigint, PRIMARY KEY): Unique identifier for each employee
   - "Name" (text): Name of the employee
   - "SalesMade" (bigint): Number of sales made by the employee
   - "AvgTransactionValue" (double precision): Average transaction value

3. TABLE: sales_transactions
   Description: Records sales transactions.
   Columns (use exact case as shown):
   - "TransactionID" (bigint, PRIMARY KEY): Unique identifier for each transaction
   - "DateTime" (timestamp with time zone): Date and time of the transaction
   - "Product" (text): Product sold
   - "Quantity" (bigint): Quantity of the product sold
   - "Price" (double precision): Price of the product
   - "PaymentMethod" (text): Method of payment used
   - "DiscountApplied" (text): Any discount applied
   - "EmployeeID" (bigint, FOREIGN KEY): References employee_performance("EmployeeID")
   - "CustomerID" (bigint, FOREIGN KEY): References customer_info("CustomerID")

Relationships:
- sales_transactions."EmployeeID" → employee_performance."EmployeeID"
- sales_transactions."CustomerID" → customer_info."CustomerID"

Important Notes:
1. Always use double quotes around column and table names to preserve case
2. Use CURRENT_DATE for current date in queries
3. When joining tables, use the exact column names with proper case
4. For text searches, use ILIKE for case-insensitive matching
5. Example of a correct query:
   SELECT c."Name", SUM(s."Quantity" * s."Price") as total
   FROM "customer_info" c
   JOIN "sales_transactions" s ON c."CustomerID" = s."CustomerID"
   GROUP BY c."CustomerID", c."Name"
   ORDER BY total DESC
   LIMIT 5

Remember to always use double quotes around identifiers to match the exact case in the database.
"""

def generate_sql_query(natural_language_query: str) -> Dict[str, Any]:
    """
    Convert a natural language query to a SQL query using Gemini 1.5 Flash
    
    Args:
        natural_language_query (str): The natural language query
        
    Returns:
        dict: A dictionary containing the SQL query and any additional information
    """
    # Create a prompt for the model
    prompt = f"""You are a senior database administrator. Convert the following natural language query into a PostgreSQL SQL query.
    
    {SCHEMA_INFO}
    
    Instructions:
    1. Only return valid PostgreSQL SQL
    2. Do not include any explanations or markdown formatting
    3. If the query is ambiguous, make reasonable assumptions and state them in the notes
    4. For date ranges, use current date if not specified
    
    Natural Language Query: {natural_language_query}
    
    Return the response as a JSON object with the following structure:
    {{
        "sql": "SELECT * FROM table WHERE condition",
        "assumptions": ["list", "of", "assumptions"],
        "notes": "Any additional notes about the query"
    }}
    """
    
    try:
        # Generate the response
        response = model.generate_content(prompt)
        
        # Parse the response (assuming it's in JSON format)
        try:
            # Try to parse the JSON response
            result = json.loads(response.text)
        except json.JSONDecodeError:
            # If parsing fails, try to extract JSON from markdown code blocks
            import re
            json_match = re.search(r'```json\n(.*?)\n```', response.text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
            else:
                # If all else fails, return the raw response
                result = {"sql": response.text, "assumptions": [], "notes": "Could not parse model response"}
        
        return result
        
    except Exception as e:
        return {
            "sql": "",
            "error": str(e),
            "assumptions": [],
            "notes": "Error generating SQL query"
        }

def execute_nl_query(query: str) -> Dict[str, Any]:
    """
    Execute a natural language query and return the results
    
    Args:
        query (str): Natural language query
        
    Returns:
        dict: Query results and metadata
    """
    from data import execute_query
    
    # Generate SQL from natural language
    result = generate_sql_query(query)
    
    if "sql" not in result or not result["sql"]:
        return {
            "success": False,
            "error": "Failed to generate SQL query",
            "details": result.get("error", "Unknown error")
        }
    
    try:
        # Execute the generated SQL
        query_result = execute_query(result["sql"])
        
        return {
            "success": True,
            "sql": result["sql"],
            "assumptions": result.get("assumptions", []),
            "notes": result.get("notes", ""),
            "data": query_result
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "sql": result["sql"],
            "assumptions": result.get("assumptions", []),
            "notes": "Error executing SQL query"
        }

def print_query_result(result):
    """Helper function to print query results in a formatted way"""
    print("\n" + "="*80)
    print("Generated SQL:")
    print("-"*80)
    print(result.get("sql", "No SQL generated").strip())
    
    if result.get("assumptions"):
        print("\nAssumptions:")
        for assumption in result["assumptions"]:
            print(f"- {assumption}")
            
    if result.get("notes"):
        print(f"\nNotes: {result['notes']}")
    
    if result.get("success", False):
        print("\nQuery Results (first 5 rows):")
        print("-"*80)
        if result["data"]:
            # Print column headers
            if result["data"]:
                print(" | ".join(str(key) for key in result["data"][0].keys()))
                print("-" * 80)
                # Print first 5 rows
                for row in result["data"][:5]:
                    print(" | ".join(str(val) for val in row.values()))
                if len(result["data"]) > 5:
                    print(f"... and {len(result['data']) - 5} more rows")
        else:
            print("No data returned")
    else:
        print("\nError:", result.get("error", "Unknown error"))
    print("="*80 + "\n")

def test_specific_query(query):
    """Test a specific natural language query and print the results"""
    print(f"\n{' TESTING QUERY ':=^80}", flush=True)
    print(f"Input: {query}", flush=True)
    
    try:
        # First, just generate the SQL to see the prompt
        print("\nGenerating SQL...", flush=True)
        sql_result = generate_sql_query(query)
        print("\nGenerated SQL:", flush=True)
        print("-"*80, flush=True)
        print(sql_result.get("sql", "No SQL generated").strip(), flush=True)
        
        # Then execute the full query
        print("\nExecuting query...", flush=True)
        result = execute_nl_query(query)
        print("\nQuery executed. Results:", flush=True)
        print_query_result(result)
        return result
        
    except Exception as e:
        print(f"\nError in test_specific_query: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        return {"error": str(e), "success": False}

def interactive_mode():
    """Run in interactive mode for multiple queries"""
    print("\n" + "="*80)
    print("Natural Language to SQL Query Tool")
    print("Type 'exit' or 'quit' to end the session")
    print("="*80)
    
    while True:
        query = input("\nEnter your natural language query: ").strip()
        if query.lower() in ['exit', 'quit']:
            break
            
        if not query:
            continue
            
        test_specific_query(query)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # If query is provided as command line argument
        query = " ".join(sys.argv[1:])
        test_specific_query(query)
    else:
        # Otherwise, run in interactive mode
        interactive_mode()
