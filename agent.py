import os
import json
import logging
import logging.config
import requests

from enum import Enum
from typing import Dict, List, Any, Optional, Union, Annotated, Literal, TypedDict
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.base import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage
from response_utils import format_database_response

# Import agents
from pa_agent import PersonalAssistantAgent
from analytics_agent import analytics_agent_node
from news_agent import NewsAgent, NewsArticle
from weather_agent import WeatherAgent
from horoscope_agent import HoroscopeAgent
from web_search import WebSearchAgent


# ── pre-compiled patterns (place near imports) ──────────────────────────────
# ── pre-compiled regexes using YOUR list variable names ─────────────────────
import re, logging
logger = logging.getLogger(__name__)

# keep the list literals for clarity / future editing
greetings = ["hello", "hi", "hey", "greetings",
             "good morning", "good afternoon", "good evening"]

exact_goodbyes = ["goodbye", "bye", "bye bye", "see you",
                  "see ya", "take care", "farewell"]

phrase_goodbyes = ["have a good", "have a nice"]

goodbye_words = ["goodbye", "bye", "farewell", "see you", "see ya", "take care"]

thanks_phrases = ["thank", "thanks", "appreciate"]

capability_phrases = ["what can you do", "help", "capabilities", "who are you"]

general_phrases = [
    "how are you", "what's up", "how's it going", "how do you do",
    "tell me about yourself", "who made you", "what are you", "your name"
]

# ① regex for greetings (whole-word match)
_greetings_regex = re.compile(
    r"\b(" + "|".join(map(re.escape, greetings)) + r")\b",
    re.I,
)

# ② regex for exact good-byes (entire query must match)
_exact_goodbye_regex = re.compile(
    r"^(" + "|".join(map(re.escape, exact_goodbyes)) + r")$",
    re.I,
)

# ③ regex for phrase good-byes at either end
_phrase_goodbyes_regex = re.compile(
    r"(^|\b)(" + "|".join(map(re.escape, phrase_goodbyes)) + r")(\b|$)",
    re.I,
)

# ④ regex for thanks
_thanks_regex = re.compile(
    r"\b(" + "|".join(map(re.escape, thanks_phrases)) + r")\b",
    re.I,
)

# ⑤ regex for goodbye words inside longer sentences (whole-word match)
_goodbye_words_regex = re.compile(
    r"\b(" + "|".join(map(re.escape, goodbye_words)) + r")\b",
    re.I,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Custom JSON encoder to handle Decimal types
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# Agent types enum for better type safety
class AgentType(Enum):
    SUPERVISOR = "supervisor"
    DB_QUERY = "db_query_agent"
    ANALYTICS = "analytics_agent"
    WEATHER = "weather_agent"
    PERSONAL_ASSISTANT = "personal_assistant"
    HOROSCOPE = "horoscope_agent"
    NEWS = "news_agent"
    WEB_SEARCH = "web_search_agent"  # New agent type

# Enhanced weather query analysis dataclass
@dataclass
class WeatherQueryAnalysis:
    location: Optional[str] = None
    use_geolocation: bool = False
    requested_parameters: List[str] = field(default_factory=list)
    query_type: str = "general"  # general, specific, forecast
    time_context: str = "current"  # current, today, tomorrow, week
    confidence: float = 0.0

# Agent configuration dataclass
@dataclass
class AgentConfig:
    name: str
    description: str
    keywords: List[str] = field(default_factory=list)
    priority: int = 1

# Improved state schema with better typing
class AgentState(TypedDict):
    messages: Annotated[List[Union[HumanMessage, AIMessage]], lambda x, y: x + y]
    next: Literal["supervisor", "db_query_agent", "analytics_agent", "weather_agent", "personal_assistant", "horoscope_agent", "news_agent", "web_search_agent", END]
    query: str
    selected_agent: str
    agent_output: Dict[str, Any]
    final_answer: str
    processed: bool
    context: Dict[str, Any]
    error: Optional[str]
    routing_confidence: float
    weather_analysis: Optional[WeatherQueryAnalysis]

# Load environment variables
load_dotenv()

# At the top level of the file, create a singleton supervisor
_supervisor_instance = None

def get_supervisor():
    """Get or create the supervisor instance."""
    global _supervisor_instance
    if _supervisor_instance is None:
        _supervisor_instance = SupervisorAgent()
    return _supervisor_instance

class SupervisorAgent:
    """Enhanced supervisor agent with better routing logic."""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=0,
            max_retries=3
        )
        # Remove agent initializations, keep only the routing configuration
        self.agents = {
            AgentType.DB_QUERY.value: AgentConfig(
                name="db_query_agent",
                description="Handles simple database queries and data retrieval operations",
                keywords=["data", "database", "query", "sales", "users", "records", 
                         "count", "sum", "average", "simple", "basic", "single", "direct",
                         "show me", "list", "get", "find", "lookup", "retrieve", "fetch"
                         ],
                priority=1
            ),
            # All data/analytics queries go to the analytics agent
            AgentType.ANALYTICS.value: AgentConfig(
                name="analytics_agent",
                description="Handles all data, analytics, and business intelligence questions",
                keywords=[
                    # Data and database keywords
                    "data", "database", "query", "sql", "table", "row", "column", "record", "dataset",
                    # Basic operations
                    "count", "sum", "total", "average", "avg", "max", "min", "calculate", "compute",
                    # Query patterns
                    "show me", "list", "get", "find", "lookup", "retrieve", "fetch", "display", "what is",
                    "how many", "how much", "what are", "which", "when", "where", "who", "whose",
                    # Analysis keywords
                    "analytics", "analysis", "report", "statistics", "metrics", "trend", "insight",
                    "compare", "versus", "vs", "difference", "growth", "decline", "change", "trend",
                    "performance", "efficiency", "productivity", "kpi", "metric", "measure", "ratio",
                    "how is", "why is", "what are the", "show me the", "analyze", "evaluate", "assess",
                    # Business domains
                    "sales", "revenue", "profit", "expense", "cost", "price", "inventory", "stock",
                    "customer", "client", "user", "product", "item", "order", "transaction", "purchase",
                    "quantity", "remaining", "available", "in stock", "stock level", "stock status", "inventory level",
                    # Time periods
                    "today", "yesterday", "this week", "last week", "this month", "last month",
                    "this quarter", "last quarter", "this year", "last year", "year to date", "ytd",
                    # Business intelligence
                    "business intelligence", "data analysis", "market analysis", "sales analysis",
                    "financial analysis", "performance review", "quarterly review", "annual report",
                    "year over year", "month over month", "quarter over quarter", "yoy", "mom", "qoq",
                    # Common question patterns
                    "top", "bottom", "best", "worst", "highest", "lowest", "most", "least", "between",
                    "by", "per", "for each", "group by", "sort by", "order by", "filter", "where"
                ],
                priority=1
            ),
            AgentType.WEATHER.value: AgentConfig(
                name="weather_agent",
                description="Handles weather-related queries including current conditions and forecasts",
                keywords=["weather", "temperature", "forecast", "rain", "snow", "sunny", "cloudy", 
                         "humidity", "wind", "storm", "temperature", "degrees", "hot", "cold", "chance of rain"],
                priority=1
            ),
            AgentType.HOROSCOPE.value: AgentConfig(
                name="horoscope_agent",
                description="Handles horoscope and zodiac sign related queries",
                keywords=["horoscope", "zodiac", "astrology", "star sign", "birth sign", "aries", "taurus", 
                         "gemini", "cancer", "leo", "virgo", "libra", "scorpio", "sagittarius", "capricorn", 
                         "aquarius", "pisces", "daily horoscope", "weekly horoscope", "monthly horoscope",
                         "advice for today", "daily advice", "guidance for today", "prediction",
                         "fortune", "what the stars say", "cosmic guidance", "astral forecast",
                         "lucky number", "lucky color", "compatibility", "love life", "career", "finance",
                         "health", "wellness", "spiritual"
                ],
                priority=2
            ),
            AgentType.NEWS.value: AgentConfig(
                name="news_agent",
                description="Handles news-related queries including current events, business news, and market updates",
                keywords=["news", "headlines", "current events", "latest news", "business news", "market news", 
                         "financial news", "stock market", "economy", "politics", "world news", "breaking news",
                         "update", "what's happening", "tell me about", "what's new", "recent developments",
                         "retail news", "tax updates", "budget", "economic policy", "business trends"
                ],
                priority=1
            ),
            AgentType.PERSONAL_ASSISTANT.value: AgentConfig(
                name="personal_assistant",
                description="Handles general queries and conversations",
                keywords=[],
                priority=0  # Default fallback
            ),
            AgentType.WEB_SEARCH.value: AgentConfig(
                name="web_search_agent",
                description="Handles web search queries using Google Custom Search API",
                keywords=["search", "google", "web", "find", "lookup", "internet", "online", "query", "information", "details", "data", "research", "explore", "discover", "learn", "know", "understand", "what is", "who is", "where is", "how to", "why does", "when did", "what are", "who are", "where are", "how do", "why do", "when was", "what does", "who does", "where does", "how can", "why can", "when can", "what should", "who should", "where should", "how should", "why should", "when should"],
                priority=1
            )
        }
    
    def route_query(self, query: str) -> Dict[str, Any]:
        """Enhanced routing logic with keyword matching first, then LLM, then fallback."""
        try:
            if not query or not query.strip():
                logger.warning("Empty query received")
                return {
                    "agent": AgentType.PERSONAL_ASSISTANT.value,
                    "confidence": 1.0,
                    "reasoning": "Empty query",
                    "method": "empty_query"
                }
            
            # First check if this is a general query
            if self._is_general_query(query):
                logger.info(f"Query identified as general query: {query}")
                return {
                    "agent": AgentType.PERSONAL_ASSISTANT.value,
                    "confidence": 0.9,
                    "reasoning": "General query or greeting detected",
                    "method": "direct"
                }
            
            # Try keyword-based routing first
            keyword_result = self._keyword_route_query(query)
            if keyword_result.get("confidence", 0) >= 0.8:  # High confidence match
                logger.info(f"Keyword routing to {keyword_result['agent']} with confidence {keyword_result['confidence']}")
                return keyword_result
            
            # If keyword routing has low confidence, try LLM routing
            logger.info("Attempting LLM routing due to low keyword confidence")
            llm_result = self._llm_route_query(query)
            if llm_result.get("confidence", 0) >= 0.4:
                logger.info(f"LLM routing to {llm_result['agent']} with confidence {llm_result['confidence']}")
                return llm_result
            
            # Fallback to personal assistant with logging
            logger.warning(f"Falling back to personal assistant for query: {query}")
            return {
                "agent": AgentType.PERSONAL_ASSISTANT.value,
                "confidence": 0.7,
                "reasoning": "Fallback after low confidence from both keyword and LLM routing",
                "method": "fallback"
            }
            
        except Exception as e:
            logger.error(f"Error in routing: {str(e)}")
            # Default to personal assistant on error
            return {
                "agent": AgentType.PERSONAL_ASSISTANT.value,
                "confidence": 0.6,
                "reasoning": f"Fallback to personal assistant due to error: {str(e)}",
                "method": "error_fallback"
            }
    
    def _is_general_query(self, query: str) -> bool:
        """Return True for greetings, thanks, capability questions, or good-byes."""
        q = query.strip().lower()

        if not q:
            logger.debug("Empty query → treat as general")
            return True

    # ── greetings ──────────────────────────────────────────────────────────
        if _greetings_regex.search(q):
            logger.debug("Greeting detected")
            return True

    # ── good-byes ──────────────────────────────────────────────────────────
        if _exact_goodbye_regex.fullmatch(q):
            logger.debug("Exact goodbye detected")
            return True

        if _phrase_goodbyes_regex.search(q):
            logger.debug("Phrase goodbye detected")
            return True

    # ── thanks ────────────────────────────────────────────────────────────
        if _thanks_regex.search(q):
            logger.debug("Thanks detected")
            return True

    # ── capability / meta questions ───────────────────────────────────────
        if any(p in q for p in capability_phrases):
            logger.debug("Capability question detected")
            return True

    # ── small-talk / generic conversation ─────────────────────────────────
        if any(p in q for p in general_phrases):
            logger.debug("General conversational phrase detected")
            return True

    # ── nothing matched ───────────────────────────────────────────────────
        logger.debug("Query does not match any general patterns")
        return False
    
    def _keyword_route_query(self, query: str) -> Dict[str, Any]:
        """Keyword-based routing logic with whole word matching."""
        query_lower = query.lower()
        query_words = set(query_lower.split())  # Split query into individual words for whole word matching
        
        def contains_phrase(phrases, text):
            """Check if any phrase in the list is present in the text."""
            for phrase in phrases:
                if ' ' in phrase:
                    # For multi-word phrases, check if the entire phrase is in the query
                    if phrase in text:
                        return True
                else:
                    # For single words, check if it's a complete word in the query
                    if phrase in query_words:
                        return True
            return False
        
        # Check for common horoscope-related questions
        horoscope_phrases = [
            "how will my day", "how's my day", "what does my day look like",
            "how is my day", "how's today looking", "how will today be",
            "what does today hold", "what's in store today",
            "will it be a good day", "good day for me", "lucky day", "unlucky day",
            "what should i avoid", "should i avoid", "be careful", "watch out for",
            "advice for today", "daily advice", "guidance for today", "prediction",
            "fortune", "what the stars say", "cosmic guidance", "astral forecast"
        ]
        horoscope_words = [
            "horoscope", "zodiac", "astrology", "star sign", "birth sign",
            "aries", "taurus", "gemini", "cancer", "leo", "virgo", 
            "libra", "scorpio", "sagittarius", "capricorn", "aquarius", "pisces"
        ]
        
        # Check for news-related queries
        news_phrases = [
            "latest news", "business news", "market news", "financial news",
            "stock market", "world news", "breaking news", "recent developments",
            "retail news", "tax updates", "economic policy", "business trends",
            "what's going on", "any updates on", "latest on", "recent news about",
            "any news about"
        ]
        news_words = [
            "news", "headlines", "economy", "politics", "update", "happening"
        ]
        
        # Check for database queries
        db_phrases = [
            "how much did i earn", "what's my earnings", "show me my", 
            "total for", "sum of", "count of", "report on", "data for",
            "metrics for", "statistics for", "analysis of", "monthly income",
            "quarterly report", "yearly earnings", "my sales", "my performance",
            "my numbers", "quantity of", "remaining quantity", "how many left",
            "stock level", "inventory level", "available quantity", "stock status",
            "in stock", "how much do i have", "current stock", "current inventory"
        ]
        db_words = [
            "income", "revenue", "sales", "profit", "expense", "transaction",
            "average", "financial", "customers", "employees", "product",
            "order", "sell", "stock", "buy", "remaining", "quantity",
            "shampoo", "ml", "180ml"
        ]
        
        # Check for analytics queries
        analytics_phrases = [
            "business intelligence", "data analysis", "market analysis", "sales analysis",
            "financial analysis", "performance review", "quarterly review", "annual report",
            "year over year", "month over month", "quarter over quarter", "show me the"
        ]
        analytics_words = [
            "analytics", "analysis", "report", "statistics", "metrics", "trend",
            "insight", "compare", "versus", "vs", "difference", "growth", "decline",
            "change", "performance", "efficiency", "productivity", "kpi", "metric",
            "measure", "analyze", "evaluate", "yoy", "mom", "qoq"
        ]
        
        # Check for weather queries
        weather_phrases = [
            "chance of rain", "how's the weather", "what's the weather",
            "will it rain", "is it going to rain", "do i need an umbrella"
        ]
        weather_words = [
            "weather", "temperature", "forecast", "rain", "snow", "sunny",
            "cloudy", "humidity", "wind", "storm", "degrees", "hot", "cold",
            "degree", "celcius"
        ]
        
        # Check each type of query in order of priority
        # First check for inventory-related queries
        inventory_keywords = ["quantity", "remaining", "stock", "inventory", "shampoo", "ml"]
        inventory_count = sum(1 for word in inventory_keywords if word in query_words)
        if inventory_count >= 2:  # Require at least 2 inventory-related keywords
            return {
                "agent": AgentType.ANALYTICS.value,
                "confidence": 0.95,
                "reasoning": f"Query contains {inventory_count} inventory-related keywords",
                "method": "keyword"
            }

        # Check for horoscope queries (both phrases and individual words)
        if (contains_phrase(horoscope_phrases, query_lower) or 
            any(word in query_words for word in horoscope_words)):
            return {
                "agent": AgentType.HOROSCOPE.value,
                "confidence": 0.95,
                "reasoning": "Query contains horoscope-related terms",
                "method": "keyword"
            }
        
        # Check for news queries (both phrases and individual words)
        if (contains_phrase(news_phrases, query_lower) or 
            any(word in query_words for word in news_words)):
            return {
                "agent": AgentType.NEWS.value,
                "confidence": 0.9,
                "reasoning": "Query contains news-related terms",
                "method": "keyword"
            }
        
        # Check for database queries (both phrases and individual words)
        if (contains_phrase(db_phrases, query_lower) or 
            any(word in query_words for word in db_words)):
            return {
                "agent": AgentType.ANALYTICS.value,  # Using ANALYTICS instead of DB_QUERY as per original
                "confidence": 0.95,
                "reasoning": "Query contains database-related terms",
                "method": "keyword"
            }
        
        # Check for analytics queries (both phrases and individual words)
        if (contains_phrase(analytics_phrases, query_lower) or 
            any(word in query_words for word in analytics_words)):
            return {
                "agent": AgentType.ANALYTICS.value,
                "confidence": 0.95,
                "reasoning": "Query contains analytics-related terms",
                "method": "keyword"
            }
        
        # Check for weather queries (both phrases and individual words)
        if (contains_phrase(weather_phrases, query_lower) or 
            any(word in query_words for word in weather_words)):
            return {
                "agent": AgentType.WEATHER.value,
                "confidence": 0.9,
                "reasoning": "Query contains weather-related terms",
                "method": "keyword"
            }
        
        # If no keywords match, return low confidence
        return {
            "agent": AgentType.PERSONAL_ASSISTANT.value,
            "confidence": 0.3,
            "reasoning": "No strong keyword matches found",
            "method": "keyword"
        }
    
    def _llm_route_query(self, query: str) -> Dict[str, Any]:
        """LLM-based routing for complex or ambiguous queries."""
        # Create a detailed description of all available agents
        agent_descriptions = []
        for agent_type, config in self.agents.items():
            agent_desc = {
                "name": config.name,
                "type": agent_type,
                "description": config.description,
            }
            agent_descriptions.append(agent_desc)

        prompt = f"""You are an intelligent routing system. Analyze the user query and determine the most appropriate agent to handle it.

Available Agents:
{json.dumps(agent_descriptions, indent=2)}

User Query: "{query}"

Notes:
1. For general product information that does not vary from store to store use web search

Response Format (JSON):
{{
    "agent": "agent_name",  // The selected agent's type value
    "confidence": 0.0-1.0,  // Confidence score
    "reasoning": "Explanation of why this agent was chosen",
    "method": "llm"
}}

Please analyze the query and provide your routing decision:"""

        try:
            #logger.info(f"LLM Routing Full Prompt:\n{prompt}")
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            
            # Extract JSON from response
            try:
                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0].strip()
                elif '```' in content:
                    content = content.split('```')[1].split('```')[0].strip()
                
                result = json.loads(content)
                
                # Validate the agent type
                if result["agent"] not in self.agents:
                    logger.warning(f"LLM returned invalid agent type: {result['agent']}")
                    result["agent"] = AgentType.PERSONAL_ASSISTANT.value
                
                # Ensure all required fields are present
                result.setdefault("confidence", 0.5)
                result.setdefault("reasoning", "LLM-based routing decision")
                result["method"] = "llm"
                
                return result
                logger.info(f"Routed by LLM:{result}")
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Failed to parse LLM response: {content}")
                return {
                    "agent": AgentType.PERSONAL_ASSISTANT.value,
                    "confidence": 0.3,
                    "reasoning": f"Failed to parse LLM response: {str(e)}",
                    "method": "llm_error"
                }
                
        except Exception as e:
            logger.error(f"LLM routing failed: {str(e)}")
            return {
                "agent": AgentType.PERSONAL_ASSISTANT.value,
                "confidence": 0.3,
                "reasoning": f"LLM routing failed: {str(e)}",
                "method": "llm_error"
            }
    
    def _get_agent_examples(self, agent_type: str) -> List[str]:
        """Get example queries for each agent type."""
        examples = {
            AgentType.WEATHER.value: [
                "What's the weather like today?",
                "Will it rain tomorrow?",
                "What's the temperature outside?",
                "Do I need an umbrella?",
                "How's the weather in Paris?"
            ],
            AgentType.DB_QUERY.value: [
                "Show me my sales for last month",
                "What are my top selling products?",
                "How much revenue did we make?",
                "Show me customer statistics",
                "What's my current inventory?"
            ],
            AgentType.ANALYTICS.value: [
                "What are the sales trends for the last quarter?",
                "How does our revenue compare to last year?",
                "What are the top 5 products by sales?",
                "What is the average order value?",
                "What is the customer retention rate?"
            ],
            AgentType.HOROSCOPE.value: [
                "What's my horoscope for today?",
                "How will my day be?",
                "What's in store for Libra?",
                "Should I be careful today?",
                "What do the stars say about my career?"
            ],
            AgentType.NEWS.value: [
                "What's happening in the world?",
                "Show me the latest headlines",
                "Any updates on the economy?",
                "What's new in technology?",
                "Tell me the business news"
            ],
            AgentType.PERSONAL_ASSISTANT.value: [
                "How are you today?",
                "What can you help me with?",
                "Tell me a joke",
                "What's your name?",
                "Thank you for your help"
            ]
        }
        return examples.get(agent_type, [])

# Enhanced node functions
def news_agent_node(state: AgentState) -> Dict[str, Any]:
    """News agent node to handle news-related queries."""
    try:
        # Initialize the agent in the node
        agent = NewsAgent()
        logger.info("News agent initialized")
        
        # Initialize default response
        response = "I'm sorry, I don't have any news to share at this time."
        
        # Extract query and parameters
        query = state.get("query", "").lower()
        
        try:
            # --- Country extraction logic ---
            import re
            import pycountry
            def extract_country(query):
                # First check for country names or common demonyms in the query
                query_lower = query.lower()
                
                # Check for country names in PyCountry
                for country in pycountry.countries:
                    # Check common name
                    if country.name.lower() in query_lower:
                        return country.alpha_2.lower()
                    # Check official name if exists
                    if hasattr(country, 'official_name') and getattr(country, 'official_name', '').lower() in query_lower:
                        return country.alpha_2.lower()
                
                # Common demonyms and alternative names
                country_mapping = {
                    # Americas
                    'us': 'us', 'usa': 'us', 'united states': 'us', 'united states of america': 'us',
                    'american': 'us', 'america': 'us',
                    'canada': 'ca', 'canadian': 'ca',
                    'mexico': 'mx', 'mexican': 'mx',
                    'brazil': 'br', 'brazilian': 'br',
                    'argentina': 'ar', 'argentinian': 'ar',
                    
                    # Europe
                    'uk': 'gb', 'united kingdom': 'gb', 'britain': 'gb', 'great britain': 'gb',
                    'british': 'gb', 'england': 'gb', 'scotland': 'gb', 'wales': 'gb',
                    'france': 'fr', 'french': 'fr',
                    'germany': 'de', 'german': 'de',
                    'italy': 'it', 'italian': 'it',
                    'spain': 'es', 'spanish': 'es',
                    'netherlands': 'nl', 'dutch': 'nl', 'holland': 'nl',
                    'belgium': 'be', 'belgian': 'be',
                    'switzerland': 'ch', 'swiss': 'ch',
                    'austria': 'at', 'austrian': 'at',
                    'sweden': 'se', 'swedish': 'se',
                    'norway': 'no', 'norwegian': 'no',
                    'denmark': 'dk', 'danish': 'dk',
                    'finland': 'fi', 'finnish': 'fi',
                    'russia': 'ru', 'russian': 'ru',
                    
                    # Asia
                    'india': 'in', 'indian': 'in',
                    'china': 'cn', 'chinese': 'cn',
                    'japan': 'jp', 'japanese': 'jp',
                    'south korea': 'kr', 'korean': 'kr', 'south korean': 'kr',
                    'north korea': 'kp', 'north korean': 'kp',
                    'australia': 'au', 'australian': 'au',
                    'new zealand': 'nz', 'kiwi': 'nz',
                    'singapore': 'sg', 'singaporean': 'sg',
                    'malaysia': 'my', 'malaysian': 'my',
                    'thailand': 'th', 'thai': 'th',
                    'vietnam': 'vn', 'vietnamese': 'vn',
                    'indonesia': 'id', 'indonesian': 'id',
                    'philippines': 'ph', 'filipino': 'ph',
                    'pakistan': 'pk', 'pakistani': 'pk',
                    'bangladesh': 'bd', 'bangladeshi': 'bd',
                    'sri lanka': 'lk', 'sri lankan': 'lk',
                    'nepal': 'np', 'nepalese': 'np',
                    'bhutan': 'bt', 'bhutanese': 'bt',
                    'maldives': 'mv', 'maldivian': 'mv',
                    
                    # Middle East
                    'uae': 'ae', 'united arab emirates': 'ae', 'emirati': 'ae',
                    'saudi arabia': 'sa', 'saudi': 'sa',
                    'israel': 'il', 'israeli': 'il',
                    'iran': 'ir', 'iranian': 'ir',
                    'iraq': 'iq', 'iraqi': 'iq',
                    'afghanistan': 'af', 'afghan': 'af',
                    'turkey': 'tr', 'turkish': 'tr',
                    'egypt': 'eg', 'egyptian': 'eg',
                    
                    # Africa
                    'south africa': 'za', 'south african': 'za',
                    'nigeria': 'ng', 'nigerian': 'ng',
                    'kenya': 'ke', 'kenyan': 'ke',
                    'ethiopia': 'et', 'ethiopian': 'et',
                    'egypt': 'eg', 'egyptian': 'eg',
                    'morocco': 'ma', 'moroccan': 'ma',
                    'ghana': 'gh', 'ghanaian': 'gh',
                    'tanzania': 'tz', 'tanzanian': 'tz',
                    'uganda': 'ug', 'ugandan': 'ug'
                }
                
                # Check the mapping dictionary
                for term, code in country_mapping.items():
                    if re.search(r'\b' + re.escape(term) + r'\b', query_lower):
                        return code
                        
                # Default to the configured country from environment
                return os.getenv('SOURCE_COUNTRY', 'in')
            country_code = extract_country(query)
            logger.info(f"Using country code: {country_code}")

            # Default to top news if no specific query
            # Check for topic-specific queries (e.g., pharma, technology, sports, etc.)
            topic_keywords = {
                'pharma': ['pharma', 'pharmaceutical', 'medicine', 'drug', 'healthcare', 'medical'],
                'technology': ['tech', 'technology', 'ai', 'artificial intelligence', 'startup', 'innovation'],
                'sports': ['sports', 'cricket', 'football', 'ipl', 'bcci', 'fifa'],
                'business': ['business', 'market', 'economy', 'finance', 'stocks', 'sensex', 'nifty'],
                'politics': ['politics', 'election', 'bjp', 'congress', 'modi', 'rahul gandhi']
            }
            
            # Default to user's query if no specific topic is found
            search_query = query
            
            # Check if the query contains any topic keywords
            for topic, keywords in topic_keywords.items():
                if any(keyword in query.lower() for keyword in keywords):
                    if topic == 'pharma':
                        search_query = 'pharmaceutical OR medicine OR drugs OR healthcare OR medical research'
                    elif topic == 'technology':
                        search_query = 'technology OR artificial intelligence OR startup OR innovation OR digital'
                    elif topic == 'sports':
                        search_query = 'sports OR cricket OR football OR ipl OR bcci'
                    elif topic == 'business':
                        search_query = 'business OR economy OR finance OR stocks OR market'
                    elif topic == 'politics':
                        search_query = 'politics OR election OR government OR parliament'
                    logger.info(f"Detected {topic} news request. Using search query: {search_query}")
                    break
                    
            if not query or any(term in query for term in ["latest", "top", "current", "recent"]):
                logger.info("Fetching top news")
                result = agent.get_top_news(source_country=country_code, limit=3, query=search_query if search_query != query else None)
            else:
                logger.info(f"Searching news for: {search_query}")
                result = agent.search_news(search_query, source_country=country_code, limit=3)
            
            # Process the raw API response
            if isinstance(result, dict):
                if 'error' in result:
                    error_msg = result.get('error', 'Unknown error occurred')
                    logger.error(f"News API error: {error_msg}")
                    response = f"I couldn't fetch the latest news. {error_msg}"
                    logger.error(f"Returning ERROR: {response}")
                    return {
                        "messages": [AIMessage(content=response)],
                        "agent_output": {"status": "error"},
                        "finished": True
                    }
                else:
                    articles = agent.process_news_response(result)
                    logger.info(f"Processed articles: {json.dumps(articles, indent=2)}")
                    if not articles:
                        response = "I couldn't find any news articles matching your query."
                        logger.warning(f"Returning NO_NEWS: {response} (articles length: {len(articles)})")
                        return {
                            "messages": [AIMessage(content=response)],
                            "agent_output": {"status": "no_news"},
                            "finished": True
                        }
                    else:
                        response = "📰 *Latest News Updates* 📰\n\n"
                        for i, article in enumerate(articles[:3], 1):
                            # Get and format title
                            title = str(article.get('title', 'No title')).strip() or 'Untitled Article'
                            title = ' '.join(title.split())  # Clean up extra whitespace and newlines
                            response += f"{i}. *{title}*\n"
                            
                            # Add read more link
                            url = article.get('url')
                            if url:
                                response += f"   [Read more]({url})\n"
                            
                            response += "\n"
                        
                        logger.info(f"Formatted news response with {len(articles)} articles")
                        return {
                            "messages": [AIMessage(content=response)],
                            "agent_output": {
                                "success": True,
                                "status": "success",
                                "response": response,
                                "articles": articles[:3]
                            },
                            "finished": True
                        }
        
        except Exception as e:
            logger.error(f"Error processing news response: {str(e)}\n{traceback.format_exc()}")
            response = f"I encountered an error while processing the news: {str(e)}"
        
        # Create the response dictionary
        response_dict = {
            "messages": [AIMessage(content=response)],
            "agent_output": {"status": "success"} if response != "I'm sorry, I don't have any news to share at this time." else {"status": "no_news"},
            "finished": True
        }
        return response_dict
    except Exception as e:
        logger.error(f"Error processing news response: {str(e)}\n{traceback.format_exc()}")
        return {
            "messages": [AIMessage(content=f"I encountered an error while fetching news: {str(e)}")],
            "agent_output": {"status": "error"},
            "finished": True
        }

def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """Enhanced supervisor node."""
    # Use the singleton supervisor
    supervisor = get_supervisor()
    
    query = state.get("query")
    if not query and state["messages"]:
        query = state["messages"][-1].content if hasattr(state["messages"][-1], 'content') else str(state["messages"][-1])
    
    if not query:
        logger.warning("No query found, defaulting to personal assistant")
        return {
            "next": AgentType.PERSONAL_ASSISTANT.value,
            "selected_agent": AgentType.PERSONAL_ASSISTANT.value,
            "routing_confidence": 0.9,
            "context": {
                "routing_method": "fallback",
                "routing_reasoning": "No query provided, defaulting to personal assistant"
            }
        }
    
    # Route the query
    routing_result = supervisor.route_query(query)
    selected_agent = routing_result["agent"]
    
    # Force all data-related queries to use analytics_agent
    if selected_agent == AgentType.DB_QUERY.value:
        selected_agent = AgentType.ANALYTICS.value
        routing_result["method"] = "redirected_to_analytics"
        routing_result["reasoning"] = "All data queries are now handled by the analytics agent"
    
    # Log the routing decision with more context
    logger.info(f"[Supervisor] Query: '{query}' -> Agent: {selected_agent} (confidence: {routing_result['confidence']:.2f})")
    
    # If we're routing to analytics_agent, log the full state for debugging
    if selected_agent == AgentType.ANALYTICS.value:
        logger.info(f"[Supervisor] Routing to analytics_agent with state keys: {list(state.keys())}")
        if 'messages' in state and state['messages']:
            logger.info(f"[Supervisor] Last message: {state['messages'][-1]}")
        if 'query' in state:
            logger.info(f"[Supervisor] Query in state: {state['query']}")
    
    return {
        "next": selected_agent,
        "selected_agent": selected_agent,
        "query": query,
        "routing_confidence": routing_result["confidence"],
        "context": {
            "routing_method": routing_result["method"],
            "routing_reasoning": routing_result["reasoning"]
        }
    }

def db_query_agent_node(state: AgentState) -> Dict[str, Any]:
    """Database query agent node."""
    if state.get("processed", False):
        return {"agent_output": {"message": "Query already processed", "success": False}}
    
    query = state.get("query", "")
    logger.info(f"DB Query Agent processing: {query}")
    
    try:
        result = execute_nl_query(query)
        
        logger.info(f"Generated SQL: {result.get('sql', 'N/A')}")
        if result.get("assumptions"):
            logger.info(f"Assumptions: {result['assumptions']}")
        
        if not result.get("success", False):
            error_msg = result.get("error", "Unknown database error")
            logger.error(f"DB query failed: {error_msg}")
            
            # Provide more user-friendly error messages for common database errors
            if "server closed the connection unexpectedly" in str(error_msg):
                error_msg = "I'm having trouble connecting to the database. Please try again in a moment."
            elif "connection" in str(error_msg).lower():
                error_msg = "I couldn't connect to the database. Please try again later."
                
            return {
                "agent_output": {
                    "success": False,
                    "error": error_msg,
                    "type": "database_error"
                },
                "error": error_msg
            }
        
        return {
            "agent_output": {
                "success": True,
                "data": result.get("data"),
                "sql": result.get("sql"),
                "assumptions": result.get("assumptions", []),
                "notes": result.get("notes", "")
            }
        }
        
    except Exception as e:
        error_msg = f"Unexpected error in DB agent: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return {
            "agent_output": {
                "success": False,
                "error": error_msg,
                "type": "system_error"
            },
            "error": error_msg
        }

def enhanced_weather_agent_node(state: AgentState) -> Dict[str, Any]:
    """Enhanced weather agent node with LLM-powered analysis."""
    try:
        query = state.get("query", "")
        logger.info(f"Enhanced Weather Agent processing: {query}")
        
        # Initialize the weather agent in the node
        agent = WeatherAgent()
        logger.info("Weather agent initialized")
        
        # Preprocess the query to handle common weather questions
        processed_query = preprocess_weather_query(query)
        logger.info(f"Processed weather query: {processed_query}")
        
        # If no location specified, get system location
        if not processed_query:
            location = get_system_location()
            logger.info(f"Using system location: {location}")
            processed_query = location
        
        # Get weather data
        result = agent.get_weather(processed_query)
        
        if not result.get('success'):
            error_msg = result.get('error', 'Unknown error')
            logger.error(f"Weather query failed: {error_msg}")
            return {
                "agent_output": {
                    "success": False,
                    "error": error_msg,
                    "type": result.get("type", "weather_error")
                },
                "error": error_msg
            }
        
        return {
            "agent_output": {
                "success": True,
                "data": result.get('data', {}),
                "location": result.get('location')
            }
        }
        
    except Exception as e:
        error_msg = f"Error in weather agent: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return {
            "agent_output": {
                "success": False,
                "error": error_msg,
                "type": "system_error"
            },
            "error": error_msg
        }

def get_system_location() -> str:
    """Get system location using IP geolocation."""
    logger.info("Attempting to detect system location...")
    
    geolocation_services = [
        {
            'url': 'http://ip-api.com/json/',
            'parser': lambda data: f"{data.get('city', '')}, {data.get('country', '')}" if data.get('status') == 'success' and data.get('city') else None
        },
        {
            'url': 'https://ipapi.co/json/',
            'parser': lambda data: f"{data.get('city', '')}, {data.get('country_name', '')}" if data.get('city') else None
        }
    ]
    
    for service in geolocation_services:
        try:
            response = requests.get(service['url'], timeout=10)
            if response.status_code == 200:
                data = response.json()
                location = service['parser'](data)
                if location and location.strip() != ', ':
                    logger.info(f"✅ Successfully detected location: {location}")
                    return location.strip()
        except Exception as e:
            logger.warning(f"Geolocation service failed: {str(e)}")
            continue
    
    # Fallback to environment variable or default
    default_location = os.getenv('DEFAULT_LOCATION', 'Bengaluru, India')
    logger.info(f"Using fallback location: {default_location}")
    return default_location

def preprocess_weather_query(query: str) -> str:
    """Preprocess weather queries to handle common patterns."""
    query_lower = query.lower().strip()
    
    # Common weather question patterns
    rain_patterns = [
        "will it rain",
        "is it going to rain",
        "is it raining",
        "are we getting rain",
        "chance of rain",
        "possibility of rain"
    ]
    
    # If it's a rain question without location, return empty string to use default location
    if any(pattern in query_lower for pattern in rain_patterns) and "in" not in query_lower:
        return ""
    
    # Handle "weather in [location]" pattern
    if "weather in" in query_lower:
        return query_lower.split("weather in")[1].strip()
    
    # Handle "temperature in [location]" pattern
    if "temperature in" in query_lower:
        return query_lower.split("temperature in")[1].strip()
    
    # If query contains "in [location]", extract location
    if " in " in query_lower:
        location = query_lower.split(" in ")[1].strip()
        # Verify this isn't part of another word
        if location and not any(word.startswith("in") for word in query_lower.split()):
            return location
    
    # For general weather queries without location, return empty string to use default
    weather_keywords = ["weather", "temperature", "rain", "snow", "sunny", "cloudy", "humid"]
    if any(keyword in query_lower for keyword in weather_keywords):
        return ""
        
    # If no patterns match, return the original query
    return query

def horoscope_agent_node(state: AgentState) -> Dict[str, Any]:
    """Horoscope agent node to handle horoscope related queries with enhanced question handling."""
    try:
        # Initialize the horoscope agent in the node
        agent = HoroscopeAgent()
        logger.info("Horoscope agent initialized")
        
        # Extract sign and date from query if present
        query = state["query"].lower()
        sign = None
        day = None
        
        # Look for zodiac signs in the query
        for zodiac in agent.valid_signs:
            if zodiac.lower() in query:
                sign = zodiac
                logger.info(f"Found zodiac sign in query: {sign}")
                break
        
        # Look for date references
        date_indicators = {
            'today': 'TODAY',
            'tomorrow': 'TOMORROW',
            'yesterday': 'YESTERDAY',
            'week': 'WEEK',
            'month': 'MONTH'
        }
        
        for indicator, value in date_indicators.items():
            if indicator in query:
                day = value
                logger.info(f"Found date indicator: {day}")
                break
        
        # Log the parameters being used
        logger.info(f"Fetching horoscope with sign: {sign or 'default'}, day: {day or 'TODAY'}")
        
        # Get horoscope data
        result = agent.get_horoscope(sign=sign, day=day or 'TODAY')
        logger.info(f"Horoscope API response: {json.dumps(result, indent=2) if isinstance(result, dict) else result}")
        
        # Format the response based on the type of question
        if not result.get('success'):
            error_msg = result.get('error', 'No error details provided')
            logger.error(f"Failed to fetch horoscope: {error_msg}")
            return {
                "messages": [AIMessage(content=f"I'm sorry, I couldn't fetch your horoscope. {error_msg}")],
                "agent_output": result,
                "final_answer": "I'm having trouble accessing horoscope information right now. Please try again later.",
                "processed": False,
                "next": END
            }
        
        # Extract the horoscope text
        horoscope_text = result.get('data', {}).get('horoscope_data', '')
        sign_used = result.get('sign', 'your sign').capitalize()
        date_used = result.get('data', {}).get('date', 'today')
        
        # Customize response based on question type
        if 'how will my day' in query or 'how\'s my day' in query or 'what does my day look like' in query:
            response = f"For {sign_used} on {date_used}, here's what the stars have in store for your day:\n\n{horoscope_text}"
        elif 'good day for me' in query or 'will it be a good day' in query or 'lucky day' in query:
            # Extract the general sentiment from the horoscope
            positive_indicators = ['fortunate', 'lucky', 'good', 'great', 'positive', 'favorable', 'promising']
            negative_indicators = ['challenging', 'difficult', 'tough', 'be careful', 'caution', 'avoid']
            
            if any(word in horoscope_text.lower() for word in positive_indicators):
                sentiment = "The stars are looking favorable for you today!"
            elif any(word in horoscope_text.lower() for word in negative_indicators):
                sentiment = "The stars suggest some caution might be needed today."
            else:
                sentiment = "The stars show a mix of influences today."
                
            response = f"For {sign_used} on {date_used}:\n\n{sentiment} Here's what the stars say:\n\n{horoscope_text}"
        elif 'what should i avoid' in query or 'should i avoid' in query or 'be careful' in query or 'watch out for' in query:
            # Try to extract cautionary advice
            response = f"For {sign_used} on {date_used}, here are some things to be mindful of according to your horoscope:\n\n"
            
            # Look for cautionary phrases in the horoscope text
            warnings = []
            sentences = [s.strip() for s in horoscope_text.split('.') if s.strip()]
            
            caution_words = ['avoid', 'be careful', 'caution', 'watch out', 'beware', 'steer clear', 'refrain']
            for sentence in sentences:
                if any(word in sentence.lower() for word in caution_words):
                    warnings.append(sentence)
            
            if warnings:
                response += "\n".join(warnings)
            else:
                response += "The stars don't highlight any specific warnings, but here's your daily horoscope:\n\n" + horoscope_text
        else:
            # Default response
            response = f"Here's the horoscope for {sign_used} on {date_used}:\n\n{horoscope_text}"
        
        # Create the response dictionary
        response_dict = {
            "messages": [AIMessage(content=response)],
            "agent_output": result,
            "final_answer": response,
            "processed": True,
            "next": END,
            "routing_confidence": 0.95  # High confidence for direct horoscope responses
        }
        
        # Create a log-safe version of the response for debugging
        log_safe_response = {
            "messages": [{"content": msg.content, "type": type(msg).__name__} for msg in response_dict["messages"]],
            "agent_output": response_dict["agent_output"],
            "final_answer": response_dict["final_answer"],
            "processed": response_dict["processed"],
            "next": str(response_dict["next"]),
            "routing_confidence": response_dict["routing_confidence"]
        }
        logger.info(f"Horoscope response prepared: {json.dumps(log_safe_response, indent=2)[:500]}...")  # Log first 500 chars
        return response_dict
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in horoscope_agent_node: {error_msg}\n{traceback.format_exc()}")
        
        # Create a safe error response
        error_response = {
            "success": False,
            "error": error_msg,
            "type": "horoscope_error"
        }
        
        return {
            "messages": [AIMessage(content="I encountered an error while processing your horoscope request.")],
            "agent_output": error_response,
            "final_answer": "I'm sorry, I couldn't process your horoscope request at the moment. Please try again later.",
            "processed": False,
            "next": END,
            "error": error_msg,
            "routing_confidence": 0.9
        }

def personal_assistant_node(state: AgentState) -> Dict[str, Any]:
    """Personal assistant node for general queries and conversations."""
    try:
        # Initialize the PA agent in the node
        agent = PersonalAssistantAgent()
        logger.info("Personal assistant initialized")
        
        query = state.get("query", "")
        logger.info(f"Personal Assistant processing: {query}")
        
        # Process the query with the personal assistant
        result = agent.process_query(query)
        
        return {
            "agent_output": {
                "success": result["success"],
                "response": result["response"],
                "type": result["type"]
            }
        }
        
    except Exception as e:
        error_msg = f"Error in personal assistant: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return {
            "agent_output": {
                "success": False,
                "error": error_msg,
                "type": "system_error"
            },
            "error": error_msg
        }

def enhanced_final_response_node(state: AgentState) -> Dict[str, str]:
    """Enhanced final response formatting with intelligent parameter selection."""
    try:
        agent_output = state.get("agent_output", {})
        query = state.get("query", "")
        selected_agent = state.get("selected_agent", "")
        weather_analysis = state.get("weather_analysis")
        
        # Handle errors
        if not agent_output.get("success", True):
            error_type = agent_output.get("type", "unknown")
            error_msg = agent_output.get("error", "An unknown error occurred")
            
            if error_type == "location_error":
                return {"final_answer": f"📍 {error_msg}"}
            elif error_type == "weather_api_error":
                return {"final_answer": f"🌤️ I couldn't get the weather information. {error_msg}"}
            elif error_type == "database_error":
                # For database errors, use the message we've already made user-friendly
                return {"final_answer": f"📊 Sorry couldn't fetch the data now. Please try again"}
            else:
                # For any other errors, provide a generic but friendly message
                return {"final_answer": "⚠️ I encountered an issue processing your request. Please try again in a moment."}
        
        # Format successful responses based on the selected agent
        if selected_agent == AgentType.WEATHER.value:
            return format_enhanced_weather_response(query, agent_output.get("data", {}), weather_analysis)
        elif selected_agent == AgentType.PERSONAL_ASSISTANT.value:
            # For personal assistant, check if this is a goodbye message
            response = {
                "final_answer": agent_output.get("response", "I'm not sure how to respond to that.")
            }
            # If this is a goodbye message from the personal assistant, add the exit flag
            if agent_output.get("type") == "goodbye" or agent_output.get("should_exit", False):
                response["should_exit"] = True
            return response
        elif selected_agent == AgentType.HOROSCOPE.value:
            # Check if we have a direct response in agent_output
            if 'response' in agent_output:
                return {"final_answer": agent_output['response']}
            # If not, check if we have a final_answer from the horoscope_agent_node
            elif state.get('final_answer'):
                return {"final_answer": state['final_answer']}
            else:
                return {"final_answer": "I couldn't find your horoscope. Please try again."}
        elif selected_agent == AgentType.NEWS.value:
            logger.info(f"Processing NEWS agent output: {json.dumps(agent_output, default=str, indent=2)}")
            
            # First check if we have a direct response in agent_output
            if agent_output.get("success", False) and "response" in agent_output:
                logger.info("Found direct response in agent_output")
                return {"final_answer": agent_output["response"]}
                
            # Fall back to checking messages if no direct response
            messages = agent_output.get("messages", [])
            logger.info(f"Found {len(messages)} messages in agent_output")
            
            if messages:
                msg = messages[0]
                logger.info(f"Message type: {type(msg).__name__}, content: {str(msg)[:200]}...")
                
                if hasattr(msg, "content"):
                    logger.info("Extracting content using msg.content")
                    return {"final_answer": msg.content}
                elif isinstance(msg, dict) and "content" in msg:
                    logger.info("Extracting content using msg['content']")
                    return {"final_answer": msg["content"]}
                else:
                    logger.warning(f"Unexpected message format: {type(msg)}")
                    return {"final_answer": str(msg)}
                    
            # If we get here, no news was found
            return {"final_answer": "I'm sorry, I don't have any news to share at this time."}
        elif selected_agent == AgentType.WEB_SEARCH.value:
            logger.info(f"Processing WEB_SEARCH agent output: {json.dumps(agent_output, default=str, indent=2)}")
            
            # First check if we have a direct response in agent_output
            if agent_output.get("success", False) and "response" in agent_output:
                logger.info("Found direct response in agent_output")
                return {"final_answer": agent_output["response"]}
            
            # If we have a final_answer in the state, use that
            if state.get("final_answer"):
                logger.info("Using final_answer from state")
                return {"final_answer": state["final_answer"]}
            
            # If we get here, no web search results were found
            return {"final_answer": "I'm sorry, I couldn't find any relevant information."}
        else:
            # Default to database response for any other agent
            return format_database_response(query, agent_output.get("data"))
            
    except Exception as e:
        logger.error(f"Error in final response: {str(e)}\n{traceback.format_exc()}")
        return {"final_answer": f"⚠️ An error occurred while formatting the response: {str(e)}"}

def format_enhanced_weather_response(query: str, weather_data: Dict[str, Any], analysis: WeatherQueryAnalysis) -> Dict[str, str]:
    """Format weather response using LLM based on requested parameters."""
    if not weather_data:
        return {"final_answer": "❌ No weather data available"}
    
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        
        # Create context about what user requested
        analysis_context = ""
        if analysis:
            analysis_context = f"""
Query Analysis:
- Requested parameters: {analysis.requested_parameters}
- Query type: {analysis.query_type}
- Time context: {analysis.time_context}
- Location source: {'geolocation' if analysis.use_geolocation else 'user-specified'}
"""
        
        prompt = f"""Generate a natural, conversational weather response based on the user's specific query and the available weather data.

User Query: "{query}"
{analysis_context}

Weather Data:
{json.dumps(weather_data, indent=2, ensure_ascii=False)}

Instructions:
1. Focus on what the user specifically asked for:
   - If they asked about rain/precipitation, emphasize rain conditions and probability
   - If they asked about temperature, focus on temperature details
   - If they asked about humidity, highlight humidity information
   - If they asked general weather, provide an overview

2. Be conversational and natural - avoid bullet points unless listing multiple items
3. Use appropriate emojis to make it engaging
4. Include location context naturally
5. If the user asked "Will it rain?" focus on precipitation and rain probability
6. Keep it concise but informative

Examples:
- For "Will it rain today?" → Focus on precipitation, rain conditions, and likelihood
- For "How hot is it?" → Focus on temperature and feels-like temperature  
- For "What's the weather?" → Give balanced overview
- For "Is it humid?" → Focus on humidity levels and comfort

Generate a natural response:"""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"final_answer": response.content.strip()}
        
    except Exception as e:
        logger.error(f"Error formatting enhanced weather response: {str(e)}")
        # Fallback to basic formatting
        location = weather_data.get('location', 'Unknown location')
        temp = weather_data.get('temperature', 'N/A')
        desc = ', '.join(weather_data.get('weather_descriptions', [])) or 'No description available'
        
        return {"final_answer": f"🌤️ Weather in {location}: {temp}°C, {desc}"}

def web_search_agent_node(state: AgentState) -> Dict[str, Any]:
    """Web search agent node to handle web search queries."""
    try:
        # Initialize the web search agent
        agent = WebSearchAgent()
        logger.info("Web search agent initialized")
        
        query = state.get("query", "")
        logger.info(f"Web Search Agent processing: {query}")
        
        # Process the search query
        try:
            # Use handle_query instead of search
            search_response = agent.handle_query(query)
            #logger.info(f"this is the search response {search_response}")
            # Check if the response indicates an error
            if (search_response is None or 
                search_response.strip() == " " or 
                "I couldn't find any relevant information" in search_response or
                "I encountered an error" in search_response):
                error_msg = search_response if search_response else "No results found"
                logger.error(f"Web search failed: {error_msg}")
                return {
                    "messages": [AIMessage(content=error_msg)],
                    "agent_output": {
                        "success": False,
                        "error": error_msg,
                        "type": "search_error"
                    },
                    "final_answer": error_msg,
                    "processed": False,
                    "next": END
                }
            
            # Log the successful response
            logger.info(f"Web search successful, response length: {len(search_response)}")
            
            # The search_response is already formatted by Gemini, so use it directly
            return {
                "messages": [AIMessage(content=search_response)],
                "agent_output": {
                    "success": True,
                    "response": search_response,
                    "results": search_response
                },
                "final_answer": search_response,
                "processed": True,
                "next": END,
                "routing_confidence": 0.95
            }
            
        except Exception as e:
            error_msg = f"Error during web search: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return {
                "messages": [AIMessage(content="I encountered an error while searching.")],
                "agent_output": {
                    "success": False,
                    "error": error_msg,
                    "type": "search_error"
                },
                "final_answer": "I'm sorry, I encountered an error while searching for that information.",
                "processed": False,
                "next": END
            }
            
    except Exception as e:
        error_msg = f"Error in web search agent: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return {
            "messages": [AIMessage(content="I encountered a system error.")],
            "agent_output": {
                "success": False,
                "error": error_msg,
                "type": "system_error"
            },
            "final_answer": "I'm sorry, I encountered a system error while trying to search.",
            "processed": False,
            "next": END
        }

# Create enhanced workflow
def create_enhanced_workflow() -> StateGraph:
    """Create and configure the enhanced workflow graph."""
    workflow = StateGraph(AgentState)
    
    # Add nodes - only include analytics_agent, not db_query_agent
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("analytics_agent", analytics_agent_node)
    workflow.add_node("weather_agent", enhanced_weather_agent_node)
    workflow.add_node("personal_assistant", personal_assistant_node)
    workflow.add_node("horoscope_agent", horoscope_agent_node)
    workflow.add_node("news_agent", news_agent_node)
    workflow.add_node("web_search_agent", web_search_agent_node)
    workflow.add_node("final_response", enhanced_final_response_node)
    
    # Add edges - no db_query_agent edge
    workflow.add_edge("analytics_agent", "final_response")
    workflow.add_edge("weather_agent", "final_response")
    workflow.add_edge("personal_assistant", "final_response")
    workflow.add_edge("horoscope_agent", "final_response")
    workflow.add_edge("news_agent", "final_response")
    workflow.add_edge("web_search_agent", "final_response")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {
            # All data/analytics queries go to analytics_agent
            "db_query_agent": "analytics_agent",  # Redirect any remaining db_query_agent calls to analytics_agent
            "analytics_agent": "analytics_agent",
            "weather_agent": "weather_agent",
            "personal_assistant": "personal_assistant",
            "horoscope_agent": "horoscope_agent",
            "news_agent": "news_agent",
            "web_search_agent": "web_search_agent",
            END: END,
        },
    )
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    # Compile the workflow
    return workflow.compile()

# Main processing function
def process_enhanced_query(query: str) -> str:
    """Process a natural language query through the enhanced agent system."""
    logger.info(f"Processing enhanced query: {query}")
    
    # Initialize supervisor at the start
    get_supervisor()
    
    # Create workflow
    runnable = create_enhanced_workflow()
    
    # Initialize state with personal assistant as the default agent
    state = {
        "messages": [HumanMessage(content=query)],
        "next": "supervisor",
        "query": query,
        "selected_agent": AgentType.PERSONAL_ASSISTANT.value,  # Default to personal assistant
        "agent_output": {},
        "final_answer": "",
        "processed": False,
        "context": {
            "routing_method": "initial",
            "routing_reasoning": "Default initial state with personal assistant"
        },
        "error": None,
        "routing_confidence": 0.9,  # High confidence in default selection
        "weather_analysis": None
    }
    
    try:
        # Run workflow
        final_state = runnable.invoke(state)
        
        if final_state.get("final_answer"):
            logger.info(f"Successfully processed query with confidence {final_state.get('routing_confidence', 0):.2f}")
            return final_state['final_answer']
        else:
            logger.error("No final answer generated")
            return "❌ I'm sorry, I couldn't process your request."
            
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}\n{traceback.format_exc()}")
        return f"⚠️ An error occurred while processing your request: {str(e)}"

def test_horoscope_agent():
    """Test the horoscope agent with various queries."""
    print("\n" + "="*60)
    print("  Testing Horoscope Agent")
    print("="*60)
    
    # Initialize the horoscope agent
    agent = HoroscopeAgent()
    
    # Test cases
    test_cases = [
        # Basic horoscope queries
        {"query": "What's my horoscope for today?", "sign": None, "day": "TODAY"},
        {"query": "How will my day be today?", "sign": None, "day": "TODAY"},
        {"query": "What should I avoid today?", "sign": None, "day": "TODAY"},
        {"query": "Will it be a good day for me?", "sign": None, "day": "TODAY"},
        
        # Specific sign queries
        {"query": "What's the horoscope for Cancer?", "sign": "Cancer", "day": "TODAY"},
        {"query": "How will a Gemini's day be tomorrow?", "sign": "Gemini", "day": "TOMORROW"},
        
        # Date-specific queries
        {"query": "What was my horoscope yesterday?", "sign": None, "day": "YESTERDAY"},
        {"query": "What's the weekly horoscope for Leo?", "sign": "Leo", "day": "WEEK"},
    ]
    
    for case in test_cases:
        print(f"\n{'='*80}")
        print(f"TEST CASE: {case['query']}")
        print(f"Sign: {case['sign'] or 'Default'}, Day: {case['day']}")
        print("-" * 80)
        
        # Create a test state
        state = {
            "query": case["query"],
            "messages": [],
            "next": None,
            "selected_agent": "horoscope_agent",
            "agent_output": {},
            "final_answer": "",
            "processed": False,
            "context": {},
            "error": None,
            "routing_confidence": 1.0,
            "weather_analysis": None
        }
        
        try:
            # Call the horoscope agent node directly
            result = horoscope_agent_node(state)
            
            # Print the result
            if result.get("processed", False):
                print("SUCCESS:")
                print(result.get("final_answer", "No response generated"))
            else:
                print("ERROR:")
                print(result.get("error", result.get("agent_output", {}).get("error", "Unknown error occurred")))
                
        except Exception as e:
            print(f"EXCEPTION: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("  End of Horoscope Agent Tests")
    print("="*60 + "\n")

if __name__ == "__main__":
    import sys
    
    # Check for test mode
    if "--test-horoscope" in sys.argv:
        test_horoscope_agent()
    elif "--test" in sys.argv:
        # Original test queries
        test_queries = [
            # Weather agent tests
            "Will it rain today?",
            "What's the humidity in London?", 
            "Temperature in Paris",
            "How's the weather?",
            "Is it windy outside?",
            
            # Database agent tests
            "What are my sales for April?",
            "Show me the top 5 products by sales",
            
            # Horoscope agent tests
            "What's my horoscope for today?",
            "How will my day be today?",
            "What should I avoid today?",
            "What's the horoscope for Cancer?",
            
            # Personal assistant tests
            "Hello!",
            "What can you do?",
            "Tell me a joke",
            "How are you today?",
            "Thank you for your help!",
            "Who created you?",
            "What's your name?"
        ]
        
        for query in test_queries:
            print(f"\n{'='*80}")
            print(f"QUERY: {query}")
            print("-" * 80)
            response = process_enhanced_query(query)
            print(f"\nRESPONSE: {response}")
    else:
        # Interactive mode with welcome message
        print("\n" + "="*60)
        print("  Welcome to Xenie - Your Personal Assistant")
        print("  Type 'quit', 'exit', 'bye', or press Ctrl+C to exit")
        print("  Try asking about weather, horoscopes, sales data, or just say hello!")
        print("="*60 + "\n")
        
        while True:
            try:
                query = input("\nYou: ").strip()
                
                # Check for exit conditions
                if not query:
                    continue
                    
                # Check for goodbye messages
                goodbye_phrases = ['quit', 'exit', 'bye', 'goodbye', 'see you', 'see ya']
                if any(phrase in query.lower() for phrase in goodbye_phrases):
                    print("\nXenie: Goodbye! Have a wonderful day! 👋")
                    break
                    
                # Process the query and print the response
                print("\nXenie: ", end="", flush=True)
                response = process_enhanced_query(query)
                print(response)
                
                # Check if we should exit based on the response
                if hasattr(response, 'get') and response.get('should_exit'):
                    break
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! Have a great day! 👋")
                break
            except Exception as e:
                print(f"\n⚠️  An error occurred: {str(e)}")
                print("Please try again or rephrase your question.")