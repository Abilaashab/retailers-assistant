import os
from typing import Dict, Any, Optional
import logging
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PersonalAssistantAgent:
    """
    Personal Assistant Agent that acts as a friendly assistant to the retailer.
    Handles general queries and conversations when other agents aren't applicable.
    """
    
    def __init__(self):
        """Initialize the Personal Assistant Agent."""
        load_dotenv()
        self.user_name = os.getenv('DEFAULT_USER_NAME', 'Store Owner')
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,  # Slightly higher temperature for more conversational responses
            max_retries=3
        )
        self.capabilities = [
            "Analyzing store performance metrics and KPIs",
            "Providing insights on inventory management",
            "Assisting with staff scheduling and management",
            "Helping with retail business strategy and planning",
            "Analyzing customer behavior and preferences",
            "Suggesting pricing and promotion strategies",
            "Providing retail market insights and trends"
        ]
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query and generate a response.
        
        Args:
            query: The user's query string
            
        Returns:
            Dict containing the response and metadata
        """
        try:
            logger.info(f"Processing personal assistant query: {query}")
            
            # Check if this is a goodbye
            if any(phrase in query.lower() for phrase in ["bye", "goodbye", "see you", "take care"]):
                return {
                    "success": True,
                    "response": f"Goodbye! Feel free to reach out if you need any more help with your retail business. Have a great day! 👋",
                    "type": "goodbye"
                }
            
            # Check if this is a greeting
            if any(word in query.lower() for word in ["hi", "hello", "hey", "greetings"]):
                return {
                    "success": True,
                    "response": (
                        f"Hello {self.user_name}, I'm Xenie, your retail business assistant! 👋\n\n"
                        "I'm here to help you manage and grow your retail business. Here's what I can help you with:\n"
                        "• Analyzing sales performance and trends\n"
                        "• Managing inventory and stock levels\n"
                        "• Understanding customer behavior and preferences\n"
                        "• Optimizing pricing and promotions\n"
                        "• Providing business insights and recommendations\n\n"
                        "What would you like to work on today?"
                    ),
                    "type": "greeting"
                }
            
            # For other queries, use LLM to generate a response
            messages = [
                SystemMessage(content=(
                    "You are Xenie, an AI assistant specifically designed for retail store owners. "
                    "Your primary goal is to help the owner manage and grow their retail business. "
                    "Focus on providing business-focused advice, not customer service.\n\n"
                    "Key responsibilities include:\n"
                    "1. Analyzing sales and business performance\n"
                    "2. Providing inventory and supply chain recommendations\n"
                    "3. Assisting with staff management and scheduling\n"
                    "4. Offering marketing and promotion strategies\n"
                    "5. Helping with business planning and strategy\n\n"
                    "Be professional, data-driven, and focused on business outcomes. "
                    "If you don't know something, be honest and offer to help find the information. "
                    "Always maintain a business-owner perspective in your responses."
                )),
                HumanMessage(content=f"As a retail store owner, I'd like to know: {query}")
            ]
            response = self.llm.invoke(messages)
            return {
                "success": True,
                "response": response.content.strip(),
                "type": "general_response"
            }
            
        except Exception as e:
            logger.error(f"Error in personal assistant: {str(e)}")
            return {
                "success": False,
                "response": "I apologize, but I'm having trouble processing your request right now. Could you please try again or rephrase your question?",
                "type": "error"
            }
    
    def _is_goodbye(self, query: str) -> bool:
        """Check if the query is a goodbye message."""
        goodbye_phrases = [
            "goodbye", "bye", "see you", "see ya", "take care", "farewell",
            "have a good", "have a nice", "gotta go", "i'm done", "that's all"
        ]
        return any(phrase in query.lower() for phrase in goodbye_phrases)
    
    def _is_greeting(self, query: str) -> bool:
        """Check if the query is a greeting."""
        greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
        return any(greeting in query.lower() for greeting in greetings)
    
    def _generate_greeting(self) -> str:
        """Generate a personalized greeting."""
        return f"Hello {self.user_name}! I'm your personal retail assistant. How can I help you today?"
    
    def _is_capabilities_query(self, query: str) -> bool:
        """Check if the user is asking about what the assistant can do."""
        capability_keywords = ["what can you do", "help", "capabilities", "what do you do", "how can you help"]
        return any(keyword in query.lower() for keyword in capability_keywords)
    
    def _list_capabilities(self) -> str:
        """List what the assistant can do."""
        capabilities_list = "\n".join([f"• {cap}" for cap in self.capabilities])
        return f"I can help you with various retail-related tasks, including:\n{capabilities_list}\n\nHow may I assist you today, {self.user_name}?"
    
    def _is_out_of_scope(self, response: str) -> bool:
        """Check if the response indicates the query is out of scope."""
        out_of_scope_indicators = [
            "i don't know",
            "i'm not sure",
            "i can't help with that",
            "i don't have that information",
            "i'm not able to",
            "i'm sorry, but i can't"
        ]
        return any(indicator in response.lower() for indicator in out_of_scope_indicators)
    
    def _get_out_of_scope_response(self) -> str:
        """Generate a polite out-of-scope response."""
        return f"I'm sorry, {self.user_name}, but I'm not able to assist with that specific request. I'm here to help with retail-related questions and general assistance. Could you try asking something else?"
    
    def _generate_llm_response(self, query: str) -> str:
        """Generate a response using the LLM."""
        prompt = f"""You are a friendly and helpful personal assistant for a retail business. The user's name is {self.user_name}.
        
Your name is Xenie. You are helpful, polite, and professional. You can assist with:
- Answering general questions about retail operations
- Providing business hours and store information
- Assisting with basic product inquiries
- Helping with common retail-related questions
- Engaging in friendly conversation
- Saying goodbye when the user indicates they're done

If a question is outside these areas, politely explain that you can't help with that specific request.

Current conversation:
User: {query}
Assistant:"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            return "I apologize, but I'm having trouble generating a response right now. Could you please try again?"

# Example usage
if __name__ == "__main__":
    import os
    os.environ["DEFAULT_USER_NAME"] = "John"  # For testing
    
    assistant = PersonalAssistantAgent()
    
    test_queries = [
        "Hello!",
        "What can you do?",
        "What's the weather like?",  # Should be out of scope
        "What are your business hours?",
        "Tell me a joke"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = assistant.process_query(query)
        print(f"Response: {result['response']}")
        print(f"Type: {result['type']}")
