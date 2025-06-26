import os
from typing import Dict, Any, Optional
import logging
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed from INFO to DEBUG to see debug messages
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
        self.user_name = os.getenv('DEFAULT_USER_NAME', 'Valued Customer')
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,  # Slightly higher temperature for more conversational responses
            max_retries=3
        )
        self.capabilities = [
            "Answering general questions about retail operations",
            "Providing business hours and store information",
            "Assisting with basic product inquiries",
            "Helping with common retail-related questions",
            "Engaging in friendly conversation"
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
            if self._is_goodbye(query):
                logger.info("Query identified as goodbye message")
                return {
                    "success": True,
                    "response": f"Goodbye, {self.user_name}! Have a wonderful day! 👋",
                    "type": "goodbye",
                    "should_exit": True
                }
            
            # Check if this is a greeting
            if self._is_greeting(query):
                logger.info("Query identified as greeting")
                return {
                    "success": True,
                    "response": self._generate_greeting(),
                    "type": "greeting"
                }
                
            # Check if this is a capabilities query
            if self._is_capabilities_query(query):
                logger.info("Query identified as capabilities request")
                return {
                    "success": True,
                    "response": self._list_capabilities(),
                    "type": "capabilities"
                }
            
            # Use LLM to generate a response
            logger.info("Generating LLM response for query")
            response = self._generate_llm_response(query)
            
            # Check if the response indicates the query is out of scope
            if self._is_out_of_scope(response):
                logger.info("LLM response indicates query is out of scope")
                return {
                    "success": False,
                    "response": self._get_out_of_scope_response(),
                    "type": "out_of_scope"
                }
            
            logger.info("Successfully generated response for query")
            return {
                "success": True,
                "response": response,
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
        query_lower = query.lower().strip()
        
        # More specific goodbye phrases that are less likely to match other queries
        goodbye_phrases = [
            "goodbye", 
            "bye", 
            "see you", 
            "see ya", 
            "take care", 
            "farewell",
            "i'm done",
            "that's all",
            "talk to you later",
            "catch you later",
            "until next time",
            "signing off"
        ]
        
        # Phrases that should only match at the start/end of the query
        boundary_phrases = [
            "have a good",
            "have a nice"
        ]
        
        logger.debug(f"Checking if '{query_lower}' is a goodbye message")
        
        # Check exact matches first
        if query_lower in ["bye", "goodbye", "exit", "quit"]:
            logger.debug(f"Exact goodbye match: '{query_lower}'")
            return True
            
        # Check for boundary phrases (must be at start or end)
        for phrase in boundary_phrases:
            if (query_lower.startswith(phrase) or 
                query_lower.endswith(phrase)):
                logger.debug(f"Matched boundary phrase: '{phrase}' in query: '{query_lower}'")
                return True
        
        # Check for other phrases as whole words only
        for phrase in goodbye_phrases:
            # Check exact match
            if query_lower == phrase:
                logger.debug(f"Exact phrase match: '{phrase}'")
                return True
                
            # Check as complete words (surrounded by word boundaries or string boundaries)
            import re
            if re.search(rf'(^|\s){re.escape(phrase)}(\s|$)', query_lower):
                logger.debug(f"Whole word match for phrase: '{phrase}' in query: '{query_lower}'")
                return True
                
        logger.debug(f"No goodbye phrases matched in: '{query_lower}'")
        return False
    
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
