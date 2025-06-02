import os
import json
import decimal
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from data import execute_query  # Import the execute_query function from data.py

# Custom JSON encoder to handle Decimal types
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            return float(obj)  # Convert Decimal to float for JSON serialization
        return super().default(obj)

# Load environment variables
load_dotenv()

# Configure Google Generative AI with API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not found. Please set it in your .env file.")

genai.configure(api_key=GOOGLE_API_KEY)

# Debug flag: set via environment variable NLQ_DEBUG or fallback to False
DEBUG = os.getenv("NLQ_DEBUG", "False").lower() in ("1", "true", "yes")

# Initialize Gemini model
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
   - "LoyaltyCard" (text): Yes/No presence or absence of loyalty card
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
   - "PaymentMethod" (text): Method of payment used which can be cash/card/UPI
   - "DiscountApplied" (text): Discount percentage applied (e.g., '10%', '15% off'). When calculating actual discount amounts, use (CAST(REPLACE(REPLACE("DiscountApplied", '%', ''), ' off', '') AS FLOAT) / 100) * ("Price" * "Quantity")
   - "EmployeeID" (bigint, FOREIGN KEY): References employee_performance("EmployeeID")
   - "CustomerID" (bigint, FOREIGN KEY): References customer_info("CustomerID")

Relationships:
- sales_transactions."EmployeeID" → employee_performance."EmployeeID"
- sales_transactions."CustomerID" → customer_info."CustomerID"

Query Guidelines:
1. Always join to customer_info to get customer names when CustomerID is involved
2. Always join to employee_performance to get employee names when EmployeeID is involved
3. For discount calculations, use: 
   - To get the discount percentage as a number: CAST(REPLACE(REPLACE("DiscountApplied", '%', ''), ' off', '') AS FLOAT)
   - To calculate discount amount: (CAST(REPLACE(REPLACE("DiscountApplied", '%', ''), ' off', '') AS FLOAT) / 100) * ("Price" * "Quantity")
   - To get final amount after discount: ("Price" * "Quantity") - ((CAST(REPLACE(REPLACE("DiscountApplied", '%', ''), ' off', '') AS FLOAT) / 100) * ("Price" * "Quantity"))
4. Prefer human-readable names over IDs in results
5. Use table aliases for better readability (e.g., 'c' for customer_info, 'e' for employee_performance, 's' for sales_transactions)

Important Notes:
1. Always use double quotes around column and table names to preserve case
2. Use CURRENT_DATE for current date in queries
3. When joining tables, use the exact column names with proper case
4. For text searches, use ILIKE for case-insensitive matching
"""

def generate_sql_query(natural_language_query: str) -> Dict[str, Any]:
    """
    Convert a natural language query to a PostgreSQL query using Gemini 1.5 Flash
    
    Args:
        natural_language_query (str): The natural language query
        
    Returns:
        dict: A dictionary containing the SQL query and any additional information
    """
    try:
        # Create a prompt for the model
        prompt = f"""You are a senior database administrator. Convert the following natural language query into a PostgreSQL SQL query.

Database Schema:
{SCHEMA_INFO}

Natural Language Query: "{natural_language_query}"

Instructions:
1. Use the exact column and table names as shown in the schema (case-sensitive, use double quotes)
2. For transaction amounts, calculate the total as ("Price" * "Quantity") when the query refers to 'amount', 'total', or 'value' of a transaction
3. Use proper JOINs to include related information (e.g., customer names, employee names)
4. Always use table aliases (e.g., 's' for sales_transactions, 'c' for customer_info, 'e' for employee_performance)
5. Format dates using TO_CHAR() for better readability
6. Use ILIKE for case-insensitive text searches
7. For count queries, use COUNT(*) and alias it as 'transaction_count'

Return a JSON object with the following structure:
{{
    "sql": "the generated SQL query",
    "assumptions": ["list any assumptions made"],
    "notes": "any additional notes"
}}

Example Response:
{{
    "sql": "SELECT COUNT(*) AS \"transaction_count\" FROM \"sales_transactions\" WHERE (\"Price\" * \"Quantity\") > 10",
    "assumptions": ["The user wants to count transactions where the total amount (Price * Quantity) is greater than 10"],
    "notes": "Used exact column names with double quotes for case sensitivity"
}}"""

        if DEBUG:
            print("\n=== GENERATING SQL QUERY ===")
            print(f"Query: {natural_language_query}")
            print("Sending to Gemini model...")

        # Generate the SQL query using Gemini
        response = model.generate_content(prompt)
        
        if DEBUG:
            print("\n=== GEMINI RESPONSE ===")
            print(response.text)
            print("=" * 40 + "\n")
        
        # Extract JSON from the response
        try:
            # Try to find JSON in the response
            json_start = response.text.find('{')
            json_end = response.text.rfind('}') + 1
            json_str = response.text[json_start:json_end].strip()
            
            # Parse the JSON
            result = json.loads(json_str)
            
            # Ensure required fields exist
            if 'sql' not in result:
                raise ValueError("Response missing 'sql' field")
                
            # Add success flag
            result['success'] = True
            
            # Add default values for optional fields
            if 'assumptions' not in result:
                result['assumptions'] = []
            if 'notes' not in result:
                result['notes'] = ""
            
            if DEBUG:
                print("\n=== PARSED RESULT ===")
                print(f"SQL: {result['sql']}")
                print(f"Assumptions: {result['assumptions']}")
                print(f"Notes: {result['notes']}")
                print("=" * 40 + "\n")
            
            return result
            
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse JSON response from model: {str(e)}"
            if DEBUG:
                print(f"\n[ERROR] {error_msg}")
                print(f"Response text: {response.text}")
            return {
                'success': False,
                'error': error_msg,
                'raw_response': response.text
            }
            
    except Exception as e:
        error_msg = f"Error generating SQL query: {str(e)}"
        if DEBUG:
            print(f"\n[ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
        return {
            'success': False,
            'error': error_msg
        }

def execute_nl_query(natural_language_query: str) -> Dict[str, Any]:
    """
    Execute a natural language query against the database.
    
    Args:
        natural_language_query (str): The natural language query
        
    Returns:
        Dict: The query results and metadata
    """
    if DEBUG:
        print(f"\n[DEBUG] Starting query execution for: {natural_language_query}")
    
    # First generate the SQL query
    sql_result = generate_sql_query(natural_language_query)
    
    if DEBUG:
        print("\n=== SQL GENERATION RESULT OBJECT ===")
        print(json.dumps(sql_result, indent=2, ensure_ascii=False, cls=DecimalEncoder))
        print("=" * 40 + "\n")
    
    # Check if SQL generation was successful
    if not sql_result.get("success", False):
        return {
            "success": False,
            "error": sql_result.get("error", "Failed to generate SQL query"),
            "assumptions": sql_result.get("assumptions", []),
            "notes": sql_result.get("notes", "SQL generation failed")
        }
    
    sql_query = sql_result["sql"]
    
    try:
        # Execute the actual query using the Supabase connection
        if DEBUG:
            print(f"[DEBUG] Executing PostgreSQL query: {sql_query}")
        results = execute_query(sql_query)
        
        if not results:
            return {
                "success": True,
                "sql": sql_query,
                "assumptions": sql_result.get("assumptions", []),
                "notes": "Query executed successfully but returned no results",
                "data": []
            }
            
        # Convert datetime objects to strings if present
        for row in results:
            for key, value in row.items():
                if isinstance(value, (datetime, str)) and ('date' in key.lower() or 'time' in key.lower()):
                    try:
                        # Convert string to datetime and back to standard format
                        dt = datetime.fromisoformat(value) if isinstance(value, str) else value
                        row[key] = dt.strftime('%Y-%m-%d %H:%M:%S')
                    except (ValueError, TypeError):
                        pass
        
        if DEBUG:
            print("[DEBUG] Query executed successfully. Result data:")
            print(json.dumps(results, indent=2, ensure_ascii=False, cls=DecimalEncoder))
        
        return {
            "success": True,
            "sql": sql_query,
            "assumptions": sql_result.get("assumptions", []),
            "notes": sql_result.get("notes", "Query executed successfully"),
            "data": results
        }
        
    except Exception as e:
        if DEBUG:
            print("[DEBUG] Error executing query:")
            print(json.dumps({"error": str(e), "sql": sql_query}, indent=2, ensure_ascii=False, cls=DecimalEncoder))
        return {
            "success": False,
            "error": f"Error executing query: {str(e)}",
            "sql": sql_query,
            "assumptions": sql_result.get("assumptions", []),
            "notes": "Query execution failed"
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
    print(f"\n{' TESTING QUERY ':=^80}")
    print(f"Input: {query}")
    
    try:
        # First, just generate the SQL to see the prompt
        print("\nGenerating SQL...")
        sql_result = generate_sql_query(query)
        print("\nGenerated SQL:")
        print("-"*80)
        print(sql_result.get("sql", "No SQL generated").strip())
        
        # Then execute the full query
        print("\nExecuting query...")
        result = execute_nl_query(query)
        print("\nQuery executed. Results:")
        print_query_result(result)
        return result
        
    except Exception as e:
        print(f"\nError in test_specific_query: {str(e)}")
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
