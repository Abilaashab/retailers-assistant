import os
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from data import execute_query  # Import the execute_query function from data.py

# Load environment variables
load_dotenv()

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

Query Guidelines:
1. Always join to customer_info to get customer names when CustomerID is involved
2. Always join to employee_performance to get employee names when EmployeeID is involved
3. Prefer human-readable names over IDs in results
4. Use table aliases for better readability (e.g., 'c' for customer_info, 'e' for employee_performance, 's' for sales_transactions)

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
        # Load environment variables
        load_dotenv()
        
        # Get API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return {
                "sql": "",
                "error": "GOOGLE_API_KEY not found in environment variables",
                "assumptions": [],
                "notes": "Configuration error"
            }
            
        # Configure Google's Gemini API
        genai.configure(api_key=api_key)
        
        # Initialize the model with a timeout
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Get current date and time with timezone info
        current_dt = datetime.now(timezone.utc)
        current_date = current_dt.strftime('%Y-%m-%d')
        current_time = current_dt.strftime('%H:%M:%S %Z')
        current_year = current_dt.year
        
        # Create a prompt for the model
        prompt = f"""You are a senior database administrator. Convert the following natural language query into a PostgreSQL SQL query.
        
        Database Schema:
        {SCHEMA_INFO}
        
        Current date: {current_date} {current_time}
        
        Important Notes:
        1. For customer-related queries, use the 'Payment_Method' column as it represents the customer
        2. Do not use any tables or columns that are not listed above
        3. For date comparisons, use the DATE() function to extract just the date part
        4. For monetary values, use the SUM() function to calculate totals
        5. Always include the DATE() function when filtering by date
        6. Return only valid PostgreSQL SQL
        
        User query: "{natural_language_query}"
        
        Respond with a JSON object containing:
        - sql: The SQL query
        - assumptions: Any assumptions made
        - notes: Any additional notes about the query
        
        Example response for "Who is my biggest customer?":
        {{
            "sql": "SELECT Payment_Method, SUM(Price) as total_spent FROM sales_transactions GROUP BY Payment_Method ORDER BY total_spent DESC LIMIT 1",
            "assumptions": ["Using Payment_Method to identify customers"],
            "notes": "Grouping by Payment_Method to find the customer with highest total spending"
        }}
        """
        
        # Generate the response with a timeout
        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 2048,
                "temperature": 0,
            },
            request_options={
                "timeout": 60  # 60 seconds timeout
            }
        )
        
        if not response or not hasattr(response, 'text'):
            return {
                "sql": "",
                "error": "Empty or invalid response from model",
                "assumptions": [],
                "notes": "Could not generate SQL query"
            }
            
        # Extract text from response
        response_text = response.text.strip()
        
        # Parse the response (assuming it's in JSON format)
        try:
            # Clean the response text
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
                
            # Parse the JSON
            result = json.loads(response_text)
            print("=== PARSED GEMINI JSON ===")
            print(result)
            print("=========================")

            # Patch: Always return a dict with 'success': True if 'sql' is present
            if isinstance(result, dict) and "sql" in result:
                return {
                    "success": True,
                    "sql": result["sql"],
                    "assumptions": result.get("assumptions", []),
                    "notes": result.get("notes", [])
                }
            else:
                return {
                    "success": False,
                    "error": "The Gemini response did not contain a valid SQL query.",
                    "assumptions": result.get("assumptions", []) if isinstance(result, dict) else [],
                    "notes": result.get("notes", []) if isinstance(result, dict) else []
                }
                
            return result
            
        except json.JSONDecodeError as e:
            return {
                "sql": "",
                "error": f"Failed to parse model response as JSON: {str(e)}\nResponse: {response_text}",
                "assumptions": [],
                "notes": "Could not parse model response"
            }
            
    except Exception as e:
        return {
            "sql": "",
            "error": f"Error generating SQL query: {str(e)}",
            "assumptions": [],
            "notes": "Error generating SQL query"
        }

def execute_nl_query(natural_language_query: str) -> Dict[str, Any]:
    """
    Execute a natural language query against the database.
    
    Args:
        natural_language_query (str): The natural language query
        
    Returns:
        Dict: The query results and metadata
    """
    print(f"\n[DEBUG] Starting query execution for: {natural_language_query}")
    
    # First generate the SQL query
    sql_result = generate_sql_query(natural_language_query)
    
    # Print the full SQL generation result for debugging
    print("\n=== SQL GENERATION RESULT OBJECT ===")
    print(json.dumps(sql_result, indent=2, ensure_ascii=False))
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
        
        return {
            "success": True,
            "sql": sql_query,
            "assumptions": sql_result.get("assumptions", []),
            "notes": sql_result.get("notes", "Query executed successfully"),
            "data": results
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[ERROR] Exception in execute_nl_query: {error_details}")
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
