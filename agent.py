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
    next: Literal["supervisor", "db_query_agent", "weather_agent", "personal_assistant", END]
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
                description="Provides current weather information and forecasts for any location",
                keywords=["weather", "temperature", "rain", "snow", "sunny", "cloudy", "forecast", 
                         "hot", "cold", "humid", "°c", "°f", "degrees", "climate", "conditions"],
                priority=2
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
        """Use LLM for routing decisions."""
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
3. For all other general queries, conversations, greetings, or when unsure, use personal_assistant
4. Weather queries don't need to mention specific locations - the weather agent handles location detection
5. If the query is a greeting, general question, or doesn't fit other categories, use personal_assistant
6. Be conservative - if you're not highly confident (>=0.8) about db_query_agent or weather_agent, default to personal_assistant

Respond with JSON:
{{
    "agent": "agent_name",  // One of: db_query_agent, weather_agent, or personal_assistant
    "confidence": 0.8,  // Confidence score between 0 and 1
    "reasoning": "Brief explanation of why this agent was chosen"
}}"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            
            # Extract JSON
            if content.startswith('```json'):
                content = content.split('```json')[1].split('```')[0].strip()
            elif content.startswith('```'):
                content = content.split('```')[1].split('```')[0].strip()
            
            result = json.loads(content)
            
            if result["agent"] not in self.agents:
                result["agent"] = AgentType.DB_QUERY.value
            
            result.setdefault("confidence", 0.5)
            result.setdefault("reasoning", "LLM-based routing")
            result["method"] = "llm"
            
            return result
            
        except Exception as e:
            logger.error(f"LLM routing failed: {str(e)}")
            return {
                "agent": AgentType.DB_QUERY.value,
                "confidence": 0.1,
                "reasoning": f"LLM routing failed: {str(e)}",
                "method": "fallback"
            }

# Enhanced node functions
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
                return {"final_answer": f"📊 I couldn't access the database. {error_msg}"}
            else:
                return {"final_answer": f"⚠️ {error_msg}"}
        
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
    """Format database response using LLM."""
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        
        prompt = f"""Generate a concise, natural language answer for the user query based on the data provided.

User Query: {query}

Data:
{json.dumps(data, indent=2, ensure_ascii=False, cls=DecimalEncoder)}

Instructions:
- If data is empty or None, politely indicate no results were found
- Summarize key findings in a user-friendly way
- Use appropriate emojis and formatting
- Keep the response concise but informative

Response:"""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"final_answer": response.content.strip()}
        
    except Exception as e:
        logger.error(f"Error formatting database response: {str(e)}")
        return {"final_answer": f"📊 I found some data but couldn't format it properly: {str(e)}"}

# Create enhanced workflow
def create_enhanced_workflow() -> StateGraph:
    """Create and configure the enhanced workflow graph."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("db_query_agent", db_query_agent_node)
    workflow.add_node("weather_agent", enhanced_weather_agent_node)
    workflow.add_node("personal_assistant", personal_assistant_node)
    workflow.add_node("final_response", enhanced_final_response_node)
    
    # Define conditional edges
    workflow.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {
            "db_query_agent": "db_query_agent",
            "weather_agent": "weather_agent",
            "personal_assistant": "personal_assistant",
            END: END,
        },
    )
    
    workflow.add_edge("db_query_agent", "final_response")
    workflow.add_edge("weather_agent", "final_response")
    workflow.add_edge("personal_assistant", "final_response")
    workflow.add_edge("final_response", END)
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
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

if __name__ == "__main__":
    import sys
    
    # Test queries for all agents
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
        
        # Personal assistant tests
        "Hello!",
        "What can you do?",
        "Tell me a joke",
        "How are you today?",
        "Thank you for your help!",
        "Who created you?",
        "What's your name?"
    ]
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
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
        print("  Try asking about weather, sales data, or just say hello!")
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