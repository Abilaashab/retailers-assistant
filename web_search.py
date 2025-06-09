import os
import requests
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from typing import List, Optional

# Configure logging with debug level
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Retrieve API key and search engine ID from environment variables
API_KEY = os.getenv("CUSTOM_SEARCH_API")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

# Retrieve Gemini API key from environment variables
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")


def web_search(query: str) -> List[dict]:
    """
    Perform a web search using Google Custom Search API.

    :param query: The search query string.
    :return: A list of search result dictionaries.
    """
    if not API_KEY or not SEARCH_ENGINE_ID:
        logger.error("Missing required environment variables: CUSTOM_SEARCH_API or SEARCH_ENGINE_ID")
        return []

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": query,
        "gl": "IN",
        "cr": "countryIN",
        "num": 5  # Limit to 5 results for better context
    }

    try:
        logger.info(f"Making request to Google Custom Search API for query: {query}")
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        results = response.json()
        
        # Log the number of results found
        items = results.get("items", [])
        logger.info(f"Found {len(items)} search results")
        
        search_results = []
        for item in items:
            result = {
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", "")
            }
            search_results.append(result)
            
            # Debug log each result
            logger.debug(f"Search Result:\nTitle: {result['title']}\nURL: {result['link']}\nSnippet: {result['snippet']}\n")
            
        return search_results
        
    except requests.RequestException as e:
        logger.error(f"Error during web search: {str(e)}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            logger.error(f"API Response: {e.response.text}")
        return []


def gemini_summarize(results: List[dict], query: str) -> str:
    """
    Summarize the search results using Gemini 1.5 Flash through LangChain.

    :param results: List of search result dictionaries.
    :param query: The original search query.
    :return: A summarized string of the search results.
    """
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0
        )
        
        if not results:
            return f"I couldn't find any relevant information about '{query}'."
        
        # Format results for the prompt
        formatted_results = "\n\n".join([
            f"Source: {result['title']}\n"
            f"URL: {result['link']}\n"
            f"Content: {result['snippet']}"
            for result in results
        ])
        
        # Log the formatted results for debugging
        logger.debug(f"Formatted search results for Gemini:\n{formatted_results}")
        
        # Create a more focused prompt that directly answers the user's query
        prompt = f"""You are a helpful assistant. The user asked: "{query}"

Here are the search results:
{formatted_results}

Based on these search results, provide a direct and clear answer to the user's question. Follow these guidelines:
1. Answer the specific question asked
2. If you find definitive information, state it clearly and cite the source URL
3. If you find conflicting information, explain the discrepancies
4. If you can't find a clear answer, explain what information is available
5. Keep your response natural and conversational, but factual

Remember to:
- Focus on answering exactly what was asked
- Include relevant source URLs when making specific claims if necessary
- Be honest about what information is or isn't available
- Answer factual questions correctly and confidently with sources mentioned for further reading
- Keep the response clear and direct"""

        messages = [HumanMessage(content=prompt)]

        logger.info("Sending request to Gemini for summarization")
        response = llm.invoke(messages)
        
        # Log Gemini's response for debugging
        logger.debug(f"Gemini response:\n{response.content}")
        
        # Log the actual response content
        summary = response.content.strip()
        #logger.info(f"Generated response: {summary}")
        
        return summary
        
    except Exception as e:
        logger.error(f"Error in Gemini summarization: {str(e)}")
        logger.exception("Full exception details:")  # This will log the full stack trace
        return f"I encountered an error while processing the search results: {str(e)}"


class WebSearchAgent:
    """
    Agent to handle web search queries using Google Custom Search API.
    """
    def __init__(self):
        """Initialize the WebSearchAgent."""
        if not API_KEY or not SEARCH_ENGINE_ID:
            logger.warning("WebSearchAgent initialized without required API credentials")

    def handle_query(self, query: str) -> str:
        """
        Handle a web search query and return the summarized results.

        :param query: The search query string.
        :return: A summarized response of search results.
        """
        try:
            # Log the incoming query
            logger.info(f"Processing web search query: {query}")
            
            # Perform the web search
            search_results = web_search(query)
            
            # Log the number of results
            logger.info(f"Retrieved {len(search_results)} search results")
            
            if not search_results:
                return "I couldn't find any relevant information. This might be due to missing API credentials or no search results found. Please try rephrasing your question."
            
            # Summarize the results using Gemini
            logger.info("Summarizing search results with Gemini")
            summary = gemini_summarize(search_results, query)
            
            # Log the length of the summary
            logger.info(f"Generated summary of length: {len(summary)}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error handling query: {str(e)}")
            logger.exception("Full exception details:")  # This will log the full stack trace
            return "I encountered an error while searching. Please try again later." 