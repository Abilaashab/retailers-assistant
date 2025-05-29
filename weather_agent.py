import os
import requests
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class WeatherAgent:
    def __init__(self):
        self.api_key = os.getenv('WEATHERSTACK_API_KEY')
        self.base_url = "http://api.weatherstack.com/current"
        
        if not self.api_key:
            raise ValueError("WEATHERSTACK_API_KEY not found in environment variables")
    
    def get_weather(self, location: str) -> Dict[str, Any]:
        """
        Get current weather for a specific location.
        
        Args:
            location (str): The city or location to get weather for
            
        Returns:
            Dict containing weather information or error details
        """
        try:
            params = {
                'access_key': self.api_key,
                'query': location
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()  # Raise exception for bad status codes
            
            data = response.json()
            
            # Check for API errors
            if 'error' in data:
                return {
                    'success': False,
                    'error': data['error'].get('info', 'Unknown error from Weatherstack API')
                }
                
            return {
                'success': True,
                'data': {
                    'location': data['location']['name'],
                    'country': data['location']['country'],
                    'temperature': data['current']['temperature'],
                    'weather_descriptions': data['current']['weather_descriptions'],
                    'humidity': data['current']['humidity'],
                    'wind_speed': data['current']['wind_speed'],
                    'wind_dir': data['current']['wind_dir'],
                    'pressure': data['current']['pressure'],
                    'precip': data['current'].get('precip', 0),
                    'feelslike': data['current'].get('feelslike'),
                    'uv_index': data['current'].get('uv_index'),
                    'visibility': data['current'].get('visibility'),
                    'is_day': data['current'].get('is_day', '').lower() == 'yes'
                }
            }
            
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f"Error connecting to Weatherstack API: {str(e)}"
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Unexpected error: {str(e)}"
            }

# Example usage
if __name__ == "__main__":
    weather_agent = WeatherAgent()
    result = weather_agent.get_weather("New York")
    print(result)
