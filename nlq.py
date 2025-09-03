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

# Debug flag: set via environment variable NLQ_DEBUG or fallback to False
DEBUG = os.getenv("NLQ_DEBUG", "False").lower() in ("1", "true", "yes")

# Initialize Gemini model
MODEL_NAME = "gemini-1.5-flash"
model = genai.GenerativeModel(MODEL_NAME)

# Database schema information with exact column cases
SCHEMA_INFO = """
Database Schema Details (Note: Column names are case-sensitive):
Table Name: batch
- id (bigint): Unique ID of the batch (primary key)
- distributor (text): Name of the distributor
- purchase_date (date): Date when the batch was purchased
- purchased_quantity (bigint): Quantity purchased in this batch
- remaining (bigint): Quantity still remaining in stock
- purchace_price (double precision): Purchase price per unit
- selling_price (bigint): Selling price per unit
- commission (double precision): Commission per unit
- expiry (date): Expiry date of the product batch

Table Name: customers
- id (bigint): Unique ID of the customer (primary key)
- name (text): Customer's name
- mobile_no (bigint): Mobile number
- address (text): Address
- age (bigint): Age of the customer
- gender (text): Gender of the customer
- avg_visit (double precision): Average visits per month
- avg_spending (double precision): Average spending per visit
- loyalty_points (bigint): Accumulated loyalty points

Table Name: employees
- id (bigint): Unique ID of the employee (primary key)
- name (text): Name of the employee
- age (bigint): Age of the employee
- gender (text): Gender
- mobile_no (bigint): Mobile number
- address (text): Address
- shift (text): Work shift (e.g., morning, evening)
- blood_group (text): Blood group of the employee

Table Name: products
- id (bigint): Unique ID of the product (primary key)
- product_name (text): Name of the product
- category (text): Product category (e.g., snacks, drinks)
- brand (text): Brand of the product
- unit (text): Unit of measure (e.g., piece, kg)
- batch_number (bigint): Batch ID the product belongs to (foreign key)
- max_stocking_quantity (bigint): Maximum stock limit
- reorder_level (bigint): Stock level at which to reorder

Table Name: transactions
- id (bigint): Unique ID of the transaction (primary key)
- order_id (bigint): Unique order number (used in Orders table)
- customer_id (bigint): Customer ID (foreign key)
- employee_id (bigint): Employee ID (foreign key)
- transaction_amount (double precision): Total bill amount
- payment_method (text): Mode of payment (e.g., cash, UPI)
- date_time (text): Date and time of transaction

Table Name: orders
- id (bigint): Unique ID of the order entry (primary key)
- order_id (bigint): Order number (foreign key to transactions.order_id)
- product_id (bigint): Product ID (foreign key)
- quantity (bigint): Quantity purchased
- price (bigint): Price per unit
- amount (bigint): Amount before tax
- igst_percentage (double precision): IGST percentage
- igst_amount (double precision): IGST value
- cgst_percentage (double precision): CGST percentage
- cgst_amount (double precision): CGST value
- sgst_percentage (double precision): SGST percentage
- sgst_amount (double precision): SGST value
- total_gst (double precision): Total GST amount
- final_amount (double precision): Total payable amount after tax

Relationships:
- transactions."employee_id" → employees."id"
- transactions."customer_id" → customers."id"
- transactions."order_id" → orders."order_id"(one-to-many)
- orders."product_id" → products."id"
- products."batch_number" → batch."id"

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
        Current year: {current_year}

        Important Notes:
        1. For customer-related queries, join with customer_info table to get customer details
        2. Do not use any tables or columns that are not listed above
        3. For date comparisons, use the DATE() function to extract just the date part
        4. For monetary values, use the SUM() function to calculate totals
        5. Always include the DATE() function when filtering by date
        6. When a date is mentioned without a year, assume the current year ({current_year}) unless specified otherwise
        7. Return only valid PostgreSQL SQL
        
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
            if DEBUG:
                print("=== PARSED GEMINI JSON ===")
                print(result)
                print("=========================")

            # Simplified: Always return a dict with all keys; set 'success' and 'error' appropriately
            output = {
                "success": False,
                "sql": "",
                "assumptions": [],
                "notes": [],
                "error": None
            }
            if isinstance(result, dict):
                output["sql"] = result.get("sql", "")
                output["assumptions"] = result.get("assumptions", [])
                output["notes"] = result.get("notes", [])
                if "sql" in result and result["sql"]:
                    output["success"] = True
                else:
                    output["error"] = "The Gemini response did not contain a valid SQL query."
            else:
                output["error"] = "The Gemini response did not contain a valid SQL query."
            return output   
            
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
