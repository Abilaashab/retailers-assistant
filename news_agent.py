import os
import json
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    """Represents a news article with relevant fields."""
    title: str
    text: str
    url: str
    source_name: str
    publish_date: str
    author: Optional[str] = None
    image_url: Optional[str] = None
    summary: Optional[str] = None

class NewsAgent:
    """Handles fetching and processing news from the World News API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the NewsAgent with API key and base URL."""
        load_dotenv()
        self.api_key = api_key or os.getenv("WORLD_NEWS_API")
        if not self.api_key:
            raise ValueError("WORLD_NEWS_API environment variable not set")
            
        self.base_url = "https://api.worldnewsapi.com"
        self.default_topic = os.getenv("DEFAULT_NEWS_TOPIC", "retail business, taxes, budget announcements in India")
        self.default_country = os.getenv("SOURCE_COUNTRY", "in")  # Default to India if not specified
        self.default_days = 3  # Default to 3 days for news freshness
        self.default_language = "en"
        self.default_sort = "publish-time"
        self.default_sort_direction = "desc"
        self.default_min_rank = 1000  # Minimum news rank (quality filter)
        
        logger.info(f"Initialized NewsAgent with default topic: {self.default_topic}")
        logger.info(f"Using source country: {self.default_country}")
        
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request to the World News API."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {"x-api-key": self.api_key}
        
        logger.info(f"Making request to: {url}")
        logger.info(f"Headers: {headers}")
        logger.info(f"Params: {params}")
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            logger.info(f"Response status: {response.status_code}")
            logger.info(f"Response headers: {response.headers}")
            
            # Log the first 500 characters of the response for debugging
            response_text = response.text
            logger.info(f"Response text (first 500 chars): {response_text[:500]}")
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making request to World News API: {str(e)}")
            return {"success": False, "error": f"Failed to fetch news: {str(e)}"}
    
    def _parse_date_range(self, days: Optional[int] = None) -> Dict[str, str]:
        """
        Generate date range parameters for the API.
        
        Args:
            days: Number of days to look back. Defaults to self.default_days (3 days)
            
        Returns:
            Dict with 'earliest-publish-date' and 'latest-publish-date' in YYYY-MM-DD format
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days if days is not None else self.default_days)
        
        # Format dates in YYYY-MM-DD format as required by the API
        return {
            "earliest-publish-date": start_date.strftime("%Y-%m-%d"),
            "latest-publish-date": end_date.strftime("%Y-%m-%d")
        }
    
    def search_news(self, query: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Search for news articles based on a query.
        
        Args:
            query: The search query string
            **kwargs: Additional query parameters (e.g., source_country, language, etc.)
            
        Returns:
            Dict containing the API response
        """
        params = {
            "text": query or self.default_topic,
            "language": kwargs.get("language", self.default_language),
            "sort": kwargs.get("sort", self.default_sort),
            "sort-direction": kwargs.get("sort_direction", self.default_sort_direction),
            "min-rank": kwargs.get("min_rank", self.default_min_rank),
            "number": min(kwargs.get("limit", 3), 100),  # Default to 3 articles per request
            "offset": kwargs.get("offset", 0),
            "source-country": kwargs.get("source_country", self.default_country),
        }
        
        # Always apply the 3-day limit unless explicitly overridden
        if not any(k in kwargs for k in ["earliest-publish-date", "latest-publish-date"]):
            date_range = self._parse_date_range(kwargs.get("days", self.default_days))
            params.update(date_range)
            logger.info(f"Search news: Applied date range: {date_range}")
        
        # Add any additional parameters
        params.update({k: v for k, v in kwargs.items() if k not in params and v is not None})
        
        return self._make_request("/search-news", params)
    
    def get_top_news(self, **kwargs) -> Dict[str, Any]:
        """
        Get top news articles.
        
        Args:
            **kwargs: Query parameters (e.g., source_country, language, query, etc.)
                query: Optional search query to filter top news
                
        Returns:
            Dict containing the API response
        """
        # Use the provided query or fall back to default topic
        search_query = kwargs.pop('query', None)
        
        params = {
            "language": kwargs.get("language", self.default_language),
            "number": min(kwargs.get("limit", 3), 100),  # Default to 3 articles per request
            "min-rank": kwargs.get("min_rank", self.default_min_rank),
            "source-country": kwargs.get("source_country", self.default_country),
            "text": search_query if search_query else self.default_topic,
        }
        
        # Always apply the 3-day limit unless explicitly overridden
        if not any(k in kwargs for k in ["earliest-publish-date", "latest-publish-date"]):
            date_range = self._parse_date_range(kwargs.get("days", self.default_days))
            params.update(date_range)
            logger.info(f"Top news: Applied date range: {date_range}")
        
        # Add any additional parameters (excluding 'query' which we've already handled)
        params.update({k: v for k, v in kwargs.items() 
                      if k not in params and v is not None and k != 'query'})
        
        return self._make_request("/search-news", params)  # Using search-news as it's more flexible
    
    def process_news_response(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process the raw API response into a more usable format.
        
        Args:
            response: The raw API response
            
        Returns:
            List of processed news articles
        """
        # The API doesn't return a 'success' field, check if we have news items instead
        articles = response.get("news", [])
        if not articles:
            logger.error(f"No news articles found in response")
            return []
            
        processed = []
        
        for article in articles:
            try:
                processed_article = {
                    "title": article.get("title", "No title"),
                    "text": article.get("text", ""),
                    "summary": article.get("summary"),
                    "url": article.get("url", ""),
                    "source_name": article.get("source", {}).get("name", "Unknown Source"),
                    "publish_date": article.get("publish_date", ""),
                    "author": article.get("authors")[0] if article.get("authors") else None,
                    "image_url": article.get("image"),
                    "language": article.get("language"),
                    "sentiment": article.get("sentiment"),
                    "relevance_score": article.get("relevance_score")
                }
                processed.append(processed_article)
            except Exception as e:
                logger.error(f"Error processing article: {str(e)}")
                continue
                
        return processed
    
    def format_news_for_display(self, articles: List[Dict[str, Any]], max_articles: int = 5) -> str:
        """
        Format news articles into a human-readable string.
        
        Args:
            articles: List of processed news articles
            max_articles: Maximum number of articles to include
            
        Returns:
            Formatted string with news summaries
        """
        if not articles:
            return "No news articles found matching your criteria."
            
        formatted = ["📰 *Latest News Updates* 📰\n"]
        
        for i, article in enumerate(articles[:max_articles], 1):
            try:
                # Safely get article details with defaults
                title = str(article.get('title', 'Untitled Article')).strip()
                if not title:
                    title = 'Untitled Article'
                    
                # Truncate title if too long
                if len(title) > 120:
                    title = title[:117] + "..."
                
                # Get source and date with safe access
                source = ""
                if 'source' in article and isinstance(article['source'], dict):
                    source_name = article['source'].get('name', '').strip()
                    if source_name:
                        source = f" ({source_name})"
                elif 'source_name' in article and article['source_name']:
                    source = f" ({article['source_name']})"
                
                date_str = ""
                if 'publish_date' in article and article['publish_date']:
                    try:
                        # Try to parse and format the date
                        date_str = f" - {article['publish_date'].split('T')[0]}"
                    except (AttributeError, IndexError, TypeError):
                        pass
                
                formatted.append(f"{i}. *{title}*{source}{date_str}")
                
                # Add URL if available
                if article.get('url'):
                    formatted.append(f"   🔗 {article['url']}")
                    
                # Add summary if available, otherwise use first 100 chars of text
                summary = article.get('summary') or article.get('text', '')[:100] + '...'
                if summary:
                    formatted.append(f"   {summary}")
                
                # Add a separator between articles
                if i < min(len(articles), max_articles):
                    formatted.append("")
                    
            except Exception as e:
                logger.error(f"Error formatting article {i}: {str(e)}")
                continue
            
        # Add source attribution
        formatted.append("\n_News powered by World News API_")
        
        return "\n".join(formatted)
    
    def get_news_summary(self, query: Optional[str] = None, **kwargs) -> str:
        """
        Get a formatted summary of news articles.
        
        Args:
            query: Optional search query
            **kwargs: Additional parameters for the API
            
        Returns:
            Formatted news summary string
        """
        # If no specific query, get top news
        if not query and not kwargs.get("source_country") and not kwargs.get("category"):
            response = self.get_top_news(**kwargs)
        else:
            response = self.search_news(query, **kwargs)
            
        articles = self.process_news_response(response)
        return self.format_news_for_display(articles, max_articles=kwargs.get("max_articles", 5))

# Example usage
if __name__ == "__main__":
    try:
        # Initialize the news agent
        news_agent = NewsAgent()
        
        # Example 1: Get top news
        print("=== Top News ===")
        print(news_agent.get_news_summary())
        
        # Example 2: Search for specific news
        print("\n=== Business News ===")
        print(news_agent.get_news_summary("business", source_country="in"))
        
        # Example 3: Get news from a specific time period
        print("\n=== Recent Tech News ===")
        print(news_agent.get_news_summary("technology", days=3, max_articles=3))
        
    except Exception as e:
        print(f"Error: {str(e)}")
