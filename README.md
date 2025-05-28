# Natural Language to SQL Query Tool

A powerful tool that converts natural language questions into SQL queries using Google's Gemini 1.5 Flash model, specifically designed to work with your Supabase database.

## Features

- Converts plain English questions into SQL queries
- Executes queries against your Supabase database
- Handles complex queries with joins, aggregations, and sorting
- Interactive command-line interface
- Case-sensitive column and table name handling

## Prerequisites

- Python 3.8+
- Supabase PostgreSQL database
- Google API key for Gemini 1.5 Flash

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   cd <repository-folder>
   ```

2. Install dependencies:
   ```bash
   pip install google-generativeai python-dotenv sqlalchemy psycopg2-binary
   ```

3. Create a `.env` file with your credentials:
   ```
   DATABASE_URL=postgresql://user:password@host:port/database
   GOOGLE_API_KEY=your-google-api-key
   ```

## Usage

### Command Line

Run a single query:
```bash
python nlq.py "Show me the top 5 customers by total spending"
```

Interactive mode:
```bash
python nlq.py
```

### Example Queries

- "Show total sales for each product category"
- "List all customers who made purchases in the last 30 days"
- "Find the employee with the highest average transaction value"
- "Show monthly sales trends for the past year"

## How It Works

1. **Natural Language Input**: User provides a question in plain English
2. **NLP Processing**: 
   - `nlq.py` sends the question to Google's Gemini 1.5 Flash model
   - Model generates SQL using the provided schema information
   - The SQL is validated and formatted
3. **Database Execution**:
   - `nlq.py` calls `data.execute_query()` with the generated SQL
   - `data.py` handles the database connection and query execution
   - Results are returned as a list of dictionaries
4. **Result Presentation**:
   - `nlq.py` formats the results for display
   - Shows the generated SQL for transparency
   - Displays query results in a clean tabular format

## Project Structure

- `nlq.py`: Handles natural language processing and SQL generation
  - Uses Google's Gemini 1.5 Flash to convert text to SQL
  - Formats and displays query results
  - Depends on `data.py` for database operations

- `data.py`: Manages all database interactions
  - Establishes connection to Supabase PostgreSQL
  - Provides `execute_query()` function used by `nlq.py`
  - Handles connection pooling and transaction management
  - Implements utility functions like `get_all()` and `get_by_id()`

- `.env`: Configuration for sensitive information (not tracked in git)
  - Database connection string
  - API keys

- `requirements.txt`: Project dependencies

### How the Files Work Together

1. `nlq.py` imports database functions from `data.py`
2. When a natural language query is received:
   - `nlq.py` generates SQL using Gemini
   - It calls `data.execute_query()` to run the SQL
   - Results are returned to `nlq.py` for display
3. This separation keeps the NLP logic separate from database concerns

## Database Schema

The tool is configured to work with the following tables:

### customer_info
- CustomerID (PK), Name, Gender, Age, LoyaltyCard, AvgSpending, VisitFrequency

### employee_performance
- EmployeeID (PK), Name, SalesMade, AvgTransactionValue

### sales_transactions
- TransactionID (PK), DateTime, Product, Quantity, Price, PaymentMethod, 
  DiscountApplied, EmployeeID (FK), CustomerID (FK)

## Error Handling

The tool provides helpful error messages for:
- Invalid SQL generation
- Database connection issues
- Query execution errors
- Missing or invalid configuration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Your License Here]

---

**Note**: Make sure to never commit your `.env` file or expose your API keys.
