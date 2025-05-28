from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get database URL from environment variable or use the provided one
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres.ismxexsmiyyruwczvylh:[YOUR-PASSWORD]@aws-0-ap-south-1.pooler.supabase.com:5432/postgres')

if not DATABASE_URL:
    raise ValueError("Please set DATABASE_URL environment variable in .env file")

# Replace postgresql with postgresql+psycopg2 for SQLAlchemy
DATABASE_URL = DATABASE_URL.replace('postgresql://', 'postgresql+psycopg2://', 1)

engine = create_engine(DATABASE_URL)
SessionLocal = scoped_session(
    sessionmaker(autocommit=False, autoflush=False, bind=engine)
)

# Create a base class for declarative models
Base = declarative_base()
metadata = MetaData()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Example model (uncomment and modify as needed)
# class YourModel(Base):
#     __tablename__ = 'your_table_name'
#     
#     id = Column(Integer, primary_key=True, index=True)
#     # Add your columns here

# Create all tables (run this once when setting up)
# Base.metadata.create_all(bind=engine)

from sqlalchemy.sql import text

def execute_query(query, params=None, fetch=True):
    """
    Execute a raw SQL query and return results.
    
    Args:
        query (str): SQL query string
        params (dict, optional): Parameters for the query
        fetch (bool): Whether to fetch results (True for SELECT, False for INSERT/UPDATE/DELETE)
    
    Returns:
        list: List of rows for SELECT queries
        int: Number of rows affected for INSERT/UPDATE/DELETE
    """
    db = SessionLocal()
    try:
        result = db.execute(text(query), params or {})
        if fetch:
            # For SELECT queries
            columns = result.keys()
            return [dict(zip(columns, row)) for row in result.fetchall()]
        else:
            # For INSERT/UPDATE/DELETE
            db.commit()
            return result.rowcount
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

def get_all(table_name):
    """Get all rows from a table"""
    return execute_query(f"SELECT * FROM {table_name}")

def get_by_id(table_name, id_value, id_column='id'):
    """Get a single row by ID"""
    result = execute_query(
        f"SELECT * FROM {table_name} WHERE {id_column} = :id_val",
        {'id_val': id_value}
    )
    return result[0] if result else None

def test_connection():
    """Test the database connection and list available tables"""
    try:
        # Test basic connection
        print("Testing database connection...")
        result = execute_query("SELECT version() AS version")
        print("✓ Successfully connected to database!")
        print(f"Database version: {result[0]['version']}")
        
        # List all tables in the public schema
        print("\nListing tables in public schema:")
        tables = execute_query("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        
        if tables:
            print("✓ Found tables:")
            for table in tables:
                print(f"- {table['table_name']}")
        else:
            print("No tables found in the public schema.")
            
    except Exception as e:
        print("❌ Error connecting to database:")
        print(str(e))
        return False
    
    return True

if __name__ == "__main__":
    test_connection()
