import json
import logging
from typing import Any, Dict, List, Union
import decimal
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

def format_database_response(query: str, data: Any) -> Dict[str, str]:
    """Format database response using LLM with Indian Rupees formatting."""
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        
        # Helper function to serialize data with proper date handling
        def serialize_data(data):
            if data is None:
                return "No data available"
                
            if isinstance(data, (list, tuple)):
                return [serialize_data(item) for item in data]
                
            if isinstance(data, dict):
                return {k: serialize_data(v) for k, v in data.items()}
                
            # Handle date and datetime objects
            if hasattr(data, 'isoformat'):
                return data.isoformat()
                
            # Handle Decimal and other numeric types
            if isinstance(data, (int, float, decimal.Decimal)):
                # Format as INR if it looks like a currency amount
                if abs(data) >= 1 and ('.' not in str(data) or str(data).endswith('.0')):
                    return f"₹{data:,.0f}"
                return f"₹{float(data):,.2f}" if data else "₹0.00"
                
            return str(data)
        
        # Serialize the data before JSON dumps
        serialized_data = serialize_data(data)
        
        prompt = f"""Generate a concise, natural language answer for the user query based on the data provided.

User Query: {query}

Data:
{json.dumps(serialized_data, indent=2, ensure_ascii=False, default=str)}

Instructions:
- If data is empty or None, politely indicate no results were found
- Format all monetary values in Indian Rupees (₹) instead of dollars
- For any currency amounts, use the format: ₹X,XXX.XX (e.g., ₹1,234.56)
- Summarize key findings in a user-friendly way
- Use appropriate emojis and formatting
- Keep the response concise but informative
- Example: Instead of $100, show as ₹100
- For date ranges, show the range in a readable format (e.g., "April 1-30, 2025")
- For daily data, summarize the key trends and highlight any significant values

Response:"""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"final_answer": response.content.strip()}
        
    except Exception as e:
        logger.error(f"Error formatting database response: {str(e)}")
        # Fallback to basic formatting
        if isinstance(data, list) and data and isinstance(data[0], dict):
            if 'remaining' in data[0]:
                return {"final_answer": f"The remaining quantity is {data[0]['remaining']} units."}
        return {"final_answer": f"Here's the data you requested: {json.dumps(data, default=str, indent=2)}"}
