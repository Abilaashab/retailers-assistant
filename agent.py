import os
import json
import logging
import requests
from typing import Dict, TypedDict, Annotated, List, Literal, Union, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from nlq import generate_sql_query, execute_nl_query
from weather_agent import WeatherAgent
from pa_agent import PersonalAssistantAgent
from horoscope_agent import HoroscopeAgent
from news_agent import NewsAgent, NewsArticle
import decimal
import traceback
from datetime import datetime

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
    WEATHER = "weather_agent"
    PERSONAL_ASSISTANT = "personal_assistant"
    HOROSCOPE = "horoscope_agent"
    NEWS = "news_agent"

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
    next: Literal["supervisor", "db_query_agent", "weather_agent", "personal_assistant", "horoscope_agent", END]
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

class EnhancedWeatherAgent:
    """Enhanced weather agent with LLM-powered location extraction and parameter selection."""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=0,
            max_retries=3
        )
        self.weather_agent = WeatherAgent()
    
    def analyze_weather_query(self, query: str) -> WeatherQueryAnalysis:
        """Use LLM to intelligently analyze weather queries."""
        
        prompt = f"""Analyze this weather query and extract the following information:

Query: "{query}"

Please respond with a JSON object containing:
{{
    "location": "specific location mentioned (city, country, etc.) or null if none",
    "use_geolocation": true/false (true if no specific location mentioned),
    "requested_parameters": ["list of specific weather parameters requested"],
    "query_type": "general|specific|forecast",
    "time_context": "current|today|tomorrow|week|specific_time",
    "confidence": 0.0-1.0
}}

Available weather parameters:
- temperature (temp, hot, cold, degrees)
- humidity (humid, moisture)
- wind (wind_speed, wind_direction, breeze)
- pressure (atmospheric_pressure, barometric)
- conditions (rain, sunny, cloudy, snow, fog, weather_description)
- precipitation (rain, snow, drizzle)
- visibility
- feels_like (apparent temperature)

Guidelines:
- If no specific location is mentioned, set use_geolocation to true
- For "Will it rain?" type questions, focus on precipitation and conditions
- For "What's the weather?" give general overview
- For "How humid is it?" focus on humidity
- Extract locations carefully - don't treat weather conditions as locations
- Be conservative with confidence scores

Examples:
- "Will it rain today?" → location: null, use_geolocation: true, requested_parameters: ["precipitation", "conditions"]
- "Temperature in Paris" → location: "Paris", use_geolocation: false, requested_parameters: ["temperature"]
- "Weather in London tomorrow" → location: "London", time_context: "tomorrow"
"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            
            # Extract JSON from response
            try:
                if content.startswith('```json'):
                    content = content.split('```json')[1].split('```')[0].strip()
                elif content.startswith('```'):
                    content = content.split('```')[1].split('```')[0].strip()
                
                result = json.loads(content)
                
                return WeatherQueryAnalysis(
                    location=result.get("location"),
                    use_geolocation=result.get("use_geolocation", False),
                    requested_parameters=result.get("requested_parameters", []),
                    query_type=result.get("query_type", "general"),
                    time_context=result.get("time_context", "current"),
                    confidence=result.get("confidence", 0.5)
                )
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {content}")
                # Fallback analysis
                return self._fallback_analysis(query)
                
        except Exception as e:
            logger.error(f"Error in LLM weather query analysis: {str(e)}")
            return self._fallback_analysis(query)
    
    def _fallback_analysis(self, query: str) -> WeatherQueryAnalysis:
        """Fallback analysis when LLM fails."""
        query_lower = query.lower()
        
        # Simple keyword detection for fallback
        has_location = any(word in query_lower for word in ['in', 'at', 'for'])
        weather_keywords = ['rain', 'temperature', 'humid', 'wind', 'weather', 'sunny', 'cloud']
        
        return WeatherQueryAnalysis(
            location=None,
            use_geolocation=True,
            requested_parameters=["conditions"] if any(kw in query_lower for kw in weather_keywords) else [],
            query_type="general",
            time_context="current",
            confidence=0.3
        )
    
    def get_enhanced_weather(self, analysis: WeatherQueryAnalysis) -> Dict[str, Any]:
        """Get weather data based on analysis."""
        try:
            # Determine location
            if analysis.use_geolocation or not analysis.location:
                location = self.get_system_location()
                if not location:
                    return {
                        "success": False,
                        "error": "Could not determine your location. Please specify a city.",
                        "type": "location_error"
                    }
            else:
                location = analysis.location
            
            logger.info(f"Getting weather for location: {location}")
            
            # Get weather data
            result = self.weather_agent.get_weather(location)
            
            if not result.get('success'):
                return {
                    "success": False,
                    "error": f"Could not get weather data: {result.get('error', 'Unknown error')}",
                    "type": "weather_api_error"
                }
            
            # Add analysis context to result
            result['analysis'] = analysis
            result['resolved_location'] = location
            
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced weather retrieval: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "type": "system_error"
            }
    
    def get_system_location(self) -> str:
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

class SupervisorAgent:
    """Enhanced supervisor agent with better routing logic."""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=0,
            max_retries=3
        )
        self.pa_agent = PersonalAssistantAgent()
        self.horoscope_agent = HoroscopeAgent()
        self.news_agent = NewsAgent()
        
        self.agents = {
            AgentType.DB_QUERY.value: AgentConfig(
                name="db_query_agent",
                description="Handles database queries, analytics, and data retrieval operations",
                keywords=["data", "database", "query", "analytics", "sales", "users", "records", 
                         "count", "sum", "average", "report", "statistics", "metrics"],
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
            )
        }
    
    def route_query(self, query: str) -> Dict[str, Any]:
        """Enhanced routing logic with LLM assistance."""
        try:
            # First, check if this is a greeting or general query
            if self._is_general_query(query):
                return {
                    "agent": AgentType.PERSONAL_ASSISTANT.value,
                    "confidence": 0.9,
                    "reasoning": "General query or greeting detected",
                    "method": "direct"
                }
                
            # Use LLM for intelligent routing
            result = self._llm_route_query(query)
            
            # If confidence is too low, default to personal assistant
            if result.get("confidence", 0) < 0.4:
                return {
                    "agent": AgentType.PERSONAL_ASSISTANT.value,
                    "confidence": 0.7,
                    "reasoning": f"Low confidence in agent selection: {result.get('reasoning', 'No reasoning provided')}",
                    "method": "fallback"
                }
                
            return result
            
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
        """Check if the query is a general conversation or greeting."""
        if not query or not query.strip():
            return True
            
        query_lower = query.lower().strip()
        
        # Check for goodbyes first
        goodbye_phrases = ["goodbye", "bye", "see you", "see ya", "take care", "farewell", "have a good", "have a nice"]
        if any(phrase in query_lower for phrase in goodbye_phrases):
            return True
            
        # Check for greetings
        greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
        if any(greeting in query_lower for greeting in greetings):
            return True
            
        # Check for thanks
        if any(phrase in query_lower for phrase in ["thank", "thanks", "appreciate"]):
            return True
            
        # Check for capabilities questions
        if any(phrase in query_lower for phrase in ["what can you do", "help", "capabilities", "who are you"]):
            return True
            
        # Check for general conversation
        general_phrases = [
            "how are you", "what's up", "how's it going", "how do you do",
            "tell me about yourself", "who made you", "what are you",
            "your name"
        ]
        if any(phrase in query_lower for phrase in general_phrases):
            return True
            
        return False
    
    def _llm_route_query(self, query: str) -> Dict[str, Any]:
        """Use LLM for routing decisions with enhanced query detection."""
        query_lower = query.lower()
        
        # Check for common horoscope-related questions
        horoscope_indicators = [
            "how will my day", "how's my day", "what does my day look like",
            "how is my day", "how's today looking", "how will today be",
            "what does today hold", "what's in store today",
            "will it be a good day", "good day for me", "lucky day", "unlucky day",
            "what should i avoid", "should i avoid", "be careful", "watch out for",
            "advice for today", "daily advice", "guidance for today", "prediction",
            "fortune", "what the stars say", "cosmic guidance", "astral forecast",
            "horoscope", "zodiac", "astrology", "star sign", "birth sign",
            # Patterns for queries about specific signs
            "day today for ", "horoscope for ", "zodiac for ", "sign for ",
            # All zodiac signs
            "aries", "taurus", "gemini", "cancer", "leo", "virgo", 
            "libra", "scorpio", "sagittarius", "capricorn", "aquarius", "pisces"
        ]
        
        # Check for news-related queries
        news_indicators = [
            "news", "headlines", "current events", "latest news", "business news", 
            "market news", "financial news", "stock market", "economy", "politics", 
            "world news", "breaking news", "update", "what's happening", "tell me about", 
            "what's new", "recent developments", "retail news", "tax updates", "budget", 
            "economic policy", "business trends", "what's going on", "any updates on",
            "latest on", "recent news about", "any news about"
        ]
        
        # Check for horoscope queries first (higher priority)
        if any(indicator in query_lower for indicator in horoscope_indicators):
            return {
                "agent": AgentType.HOROSCOPE.value,
                "confidence": 0.95,
                "reasoning": "Query contains common horoscope-related phrases",
                "method": "keyword"
            }
            
        # Then check for news queries
        if any(indicator in query_lower for indicator in news_indicators):
            return {
                "agent": AgentType.NEWS.value,
                "confidence": 0.9,
                "reasoning": "Query contains news-related keywords",
                "method": "keyword"
            }
            
        # Check for database queries
        db_query_indicators = [
            "income", "revenue", "sales", "profit", "expense", "transaction", 
            "how much did i earn", "what's my earnings", "show me my", 
            "total for", "sum of", "average", "count of", "report on",
            "data for", "metrics for", "statistics for", "analysis of",
            "financial", "monthly income", "quarterly report", "yearly earnings",
            "my sales", "my performance", "my numbers"
        ]
        if any(indicator in query_lower for indicator in db_query_indicators):
            return {
                "agent": AgentType.DB_QUERY.value,
                "confidence": 0.95,
                "reasoning": "Query contains database-related keywords",
                "method": "keyword"
            }
            
        # Check for weather queries
        weather_indicators = [
            "weather", "temperature", "forecast", "rain", "snow", "sunny", "cloudy", 
            "humidity", "wind", "storm", "degrees", "hot", "cold", "chance of rain",
            "how's the weather", "what's the weather", "will it rain", "is it going to rain",
            "do i need an umbrella"
        ]
        if any(indicator in query_lower for indicator in weather_indicators):
            return {
                "agent": AgentType.WEATHER.value,
                "confidence": 0.95,
                "reasoning": "Query contains weather-related keywords",
                "method": "keyword"
            }
            
        agent_descriptions = "\n".join([
            f"- {name}: {config.description}. Keywords: {', '.join(config.keywords) if config.keywords else 'N/A'}"
            for name, config in self.agents.items()
        ])
        
        prompt = f"""You are an intelligent routing system. Analyze the user query and determine the most appropriate agent.

Available agents:
{agent_descriptions}

User query: "{query}"

Instructions:
1. Route weather-related queries (temperature, rain, humidity, weather conditions, etc.) to weather_agent
2. Route data/database queries (sales, users, analytics, reports, etc.) to db_query_agent
3. Route ALL horoscope-related queries to horoscope_agent, including:
   - Questions about daily horoscopes (e.g., "How is my day today?")
   - Questions about specific zodiac signs (e.g., "What's the horoscope for Leo?")
   - Questions about star signs, zodiac signs, or birth signs
   - Questions asking for astrological predictions or advice
   - Questions containing any of the 12 zodiac sign names (Aries, Taurus, etc.)
   - Questions about what the stars say or cosmic guidance
4. For all other general queries, conversations, greetings, or when unsure, use personal_assistant
5. Weather queries don't need to mention specific locations - the weather agent handles location detection
6. If the query is a greeting, general question, or doesn't fit other categories, use personal_assistant
7. Be conservative - if you're not highly confident (>=0.8) about db_query_agent or weather_agent, default to personal_assistant

Respond with JSON:
{
    "agent": "agent_name",  // One of: db_query_agent, weather_agent, horoscope_agent, or personal_assistant
    "confidence": 0.8,  // Confidence score between 0 and 1
    "reasoning": "Brief explanation of why this agent was chosen"
}"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            
            # Extract JSON
            try:
                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0].strip()
                elif '```' in content:
                    content = content.split('```')[1].split('```')[0].strip()
                
                # Ensure we have valid JSON
                if not content.strip():
                    raise ValueError("Empty content after JSON extraction")
                    
                result = json.loads(content)
            except (json.JSONDecodeError, IndexError, ValueError) as e:
                logger.error(f"Failed to parse LLM response: {content}")
                raise ValueError(f"Invalid JSON response from LLM: {str(e)}")
            
            if result["agent"] not in self.agents:
                result["agent"] = AgentType.DB_QUERY.value
            
            result.setdefault("confidence", 0.5)
            result.setdefault("reasoning", "LLM-based routing")
            result["method"] = "llm"
            
            return result
            
        except Exception as e:
            logger.error(f"LLM routing failed: {str(e)}")
            return {
                "agent": AgentType.PERSONAL_ASSISTANT.value,
                "confidence": 0.1,
                "reasoning": f"LLM routing failed: {str(e)}",
                "method": "fallback"
            }

# Enhanced node functions
def news_agent_node(state: AgentState) -> Dict[str, Any]:
    """News agent node to handle news-related queries."""
    try:
        # Initialize the agent
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
                        for i, article in enumerate(articles[:3], 1):  # Limit to 3 articles
                            # Get and format title
                            title = str(article.get('title', 'No title')).strip() or 'Untitled Article'
                            title = ' '.join(title.split())  # Clean up extra whitespace and newlines
                            response += f"{i}. *{title}*\n"
                            
                            # Get and format summary/description
                            summary = article.get('summary') or article.get('text') or article.get('description')
                            if summary:
                                summary = ' '.join(str(summary).split())  # Clean up whitespace and newlines
                                # Split into sentences and rejoin with proper spacing
                                sentences = [s.strip() for s in summary.split('.') if s.strip()]
                                summary = '. '.join(sentences) + ('.' if sentences else '')
                                response += f"   {summary}\n"
                            
                            # Get and format source and date
                            source = ''
                            if 'source' in article and isinstance(article['source'], dict):
                                source = article['source'].get('name', '')
                            if not source:
                                source = article.get('source_name', '')
                            if source:
                                response += f"   Source: {source}\n"

                            pub_date = article.get('publish_date') or article.get('published_at') or article.get('date')
                            if pub_date:
                                response += f"   Published: {pub_date}\n"

                            url = article.get('url')
                            if url:
                                response += f"   Read more: {url}\n"

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
    supervisor = SupervisorAgent()
    
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
    
    logger.info(f"Query: '{query}' -> Agent: {selected_agent} (confidence: {routing_result['confidence']:.2f})")
    
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
        
        # Use enhanced weather agent
        enhanced_agent = EnhancedWeatherAgent()
        
        # Analyze the query using LLM
        analysis = enhanced_agent.analyze_weather_query(query)
        logger.info(f"Weather query analysis: location={analysis.location}, use_geolocation={analysis.use_geolocation}, parameters={analysis.requested_parameters}")
        
        # Get weather data based on analysis
        result = enhanced_agent.get_enhanced_weather(analysis)
        
        if not result.get('success'):
            error_msg = result.get('error', 'Unknown error')
            logger.error(f"Weather query failed: {error_msg}")
            return {
                "agent_output": {
                    "success": False,
                    "error": error_msg,
                    "type": result.get("type", "weather_error")
                },
                "error": error_msg,
                "weather_analysis": analysis
            }
        
        return {
            "agent_output": {
                "success": True,
                "data": result['data'],
                "location": result.get('resolved_location'),
                "analysis": analysis
            },
            "weather_analysis": analysis
        }
        
    except Exception as e:
        error_msg = f"Error in enhanced weather agent: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return {
            "agent_output": {
                "success": False,
                "error": error_msg,
                "type": "system_error"
            },
            "error": error_msg
        }

def horoscope_agent_node(state: AgentState) -> Dict[str, Any]:
    """Horoscope agent node to handle horoscope related queries with enhanced question handling."""
    try:
        # Initialize the agent
        agent = HoroscopeAgent()
        logger.info(f"Horoscope agent initialized. Default sign: {agent.default_sign}")
        
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
        query = state.get("query", "")
        logger.info(f"Personal Assistant processing: {query}")
        
        # Initialize the personal assistant
        supervisor = SupervisorAgent()
        
        # Process the query with the personal assistant
        result = supervisor.pa_agent.process_query(query)
        
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
                return {"final_answer": f"📊 {error_msg}"}
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
        logger.error(f"Data that caused the error: {str(data)[:500]}...")  # Log first 500 chars of data
        return {
            "final_answer": "📊 I found some data but encountered an issue formatting it. " \
                          "Please try rephrasing your query or ask for specific details."
        }

# Create enhanced workflow
def create_enhanced_workflow() -> StateGraph:
    """Create and configure the enhanced workflow graph."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("db_query_agent", db_query_agent_node)
    workflow.add_node("weather_agent", enhanced_weather_agent_node)
    workflow.add_node("horoscope_agent", horoscope_agent_node)
    workflow.add_node("news_agent", news_agent_node)
    workflow.add_node("personal_assistant", personal_assistant_node)
    workflow.add_node("final_response", enhanced_final_response_node)
    
    # Define conditional edges
    workflow.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {
            "db_query_agent": "db_query_agent",
            "weather_agent": "weather_agent",
            "horoscope_agent": "horoscope_agent",
            "news_agent": "news_agent",
            "personal_assistant": "personal_assistant",
            END: "final_response"
        }
    )
    
    # Add edges from agent nodes to final response
    for node in ["db_query_agent", "weather_agent", "horoscope_agent", "news_agent", "personal_assistant"]:
        workflow.add_edge(node, "final_response")
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    # Compile the workflow
    return workflow.compile()

# Main processing function
def process_enhanced_query(query: str) -> str:
    """Process a natural language query through the enhanced agent system."""
    logger.info(f"Processing enhanced query: {query}")
    
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
                goodbye_phrases = ['quit', 'exit', 'q', 'bye', 'goodbye', 'see you', 'see ya']
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