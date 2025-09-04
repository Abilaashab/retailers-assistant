import os
import json
import logging
import requests
from typing import Dict, Optional
from datetime import datetime
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class HoroscopeAgent:
    """Agent for fetching and processing horoscope data."""
    
    BASE_URL = "https://horoscope-app-api.vercel.app/api/v1/get-horoscope/daily"
    
    def __init__(self):
        self.default_sign = os.getenv("DEFAULT_USER_SIGN", "Gemini").capitalize()
        self.valid_signs = [
            'Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo',
            'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces'
        ]
    
    def get_horoscope(self, sign: Optional[str] = None, day: Optional[str] = None) -> Dict[str, any]:
        """
        Get horoscope for the specified sign and day.
        
        Args:
            sign: Zodiac sign (e.g., 'Cancer'). If None, uses default from .env
            day: Date in YYYY-MM-DD format or 'TODAY'. If None, omits the parameter
            
        Returns:
            Dictionary containing horoscope data or error information
        """
        try:
            # Use default sign if none provided
            if not sign:
                sign = self.default_sign
            else:
                # Capitalize first letter for consistency
                sign = sign.capitalize()
                
                # Validate sign
                if sign not in self.valid_signs:
                    return {
                        'success': False,
                        'error': f'Invalid zodiac sign. Please use one of: {", ".join(self.valid_signs)}',
                        'type': 'validation_error'
                    }
            
            # Prepare query parameters
            params = {'sign': sign}
            if day:
                params['day'] = day
            
            # Make API request
            response = requests.get(
                self.BASE_URL,
                params=params,
                headers={'accept': 'application/json'},
                timeout=10
            )
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse and return response
            data = response.json()
            if data.get('success', False):
                return {
                    'success': True,
                    'data': data['data'],
                    'sign': sign,
                    'day': day or 'today'
                }
            else:
                return {
                    'success': False,
                    'error': data.get('message', 'Failed to fetch horoscope'),
                    'type': 'api_error'
                }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching horoscope: {str(e)}")
            return {
                'success': False,
                'error': 'Failed to connect to horoscope service',
                'type': 'connection_error'
            }
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return {
                'success': False,
                'error': 'An unexpected error occurred',
                'type': 'unexpected_error'
            }
    
    def format_horoscope_response(self, result: Dict[str, any]) -> str:
        """Format the horoscope result into a user-friendly string."""
        if not result.get('success'):
            error_msg = result.get('error', 'Unknown error')
            return f"I couldn't fetch your horoscope. {error_msg}"
        
        data = result.get('data', {})
        sign = result.get('sign', 'your sign')
        
        return (
            f"Here's the horoscope for {sign} for {data.get('date', 'today')}:\n\n"
            f"{data.get('horoscope_data', 'No horoscope data available.')}"
        )

# Example usage
if __name__ == "__main__":
    agent = HoroscopeAgent()
    
    # Example 1: Get default horoscope (from .env)
    print("Default horoscope:")
    result = agent.get_horoscope()
    print(agent.format_horoscope_response(result))
    
    # Example 2: Get horoscope for specific sign and day
    print("\nCancer horoscope for today:")
    result = agent.get_horoscope(sign="Cancer", day="TODAY")
    print(agent.format_horoscope_response(result))
