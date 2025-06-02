import os
print(f"[DEBUG] Loaded analytics_agent.py from: {__file__}")
import logging
import json
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Import the NLQ module
from nlq import execute_nl_query

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnalyticsAgent:
    """
    Analytics Agent that handles complex analytical queries requiring multiple data points
    and advanced analysis to provide business insights and recommendations.
    """
    
    def __init__(self):
        """Initialize the Analytics Agent with the required LLM and configurations."""
        load_dotenv()
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.2,  # Lower temperature for more focused, analytical responses
            max_retries=3
        )
        self.capabilities = [
            "Analyzing retail sales performance and trends",
            "Identifying best-selling products and categories",
            "Providing inventory management recommendations",
            "Analyzing customer purchase patterns and preferences",
            "Suggesting pricing and promotion strategies",
            "Tracking foot traffic and customer engagement metrics",
            "Generating retail-specific business insights"
        ]
        
        # Initialize NLQ module
        from nlq import execute_nl_query
        self.execute_nl_query = execute_nl_query
        
    def process_analytics_query(self, query: str) -> Dict[str, Any]:
        """
        Process an analytics query by extracting parameters, fetching data, and generating insights.
        Handles multiple questions in a single query by splitting them and processing individually.
        
        Args:
            query (str): The user's query (may contain multiple questions)
            
        Returns:
            Dict[str, Any]: Response containing insights and formatted output
        """
        try:
            print("\n[DEBUG] ===== ENTERED process_analytics_query =====")
            print(f"[DEBUG] Query: {query}")
            logger.info(f"Processing analytics query: {query}")
            
            # Detect discount aggregation/ranking queries and handle them directly FIRST
            if self._is_discount_aggregation_query(query):
                logger.info("[DEBUG] Detected discount aggregation query, handling directly.")
                return self._process_discount_aggregation_query(query)
            
            # Only split into individual questions if not a discount aggregation query
            questions = self._split_into_questions(query)
            logger.info(f"Split into {len(questions)} questions: {questions}")
            
            if len(questions) > 1:
                # Process each question separately and combine results
                all_responses = []
                for q in questions:
                    logger.info(f"Processing sub-question: {q}")
                    try:
                        # Process each question as a separate query
                        result = self._process_single_query(q)
                        if result.get('success', False):
                            all_responses.append(result.get('formatted_response', ''))
                    except Exception as e:
                        logger.error(f"Error processing sub-question '{q}': {str(e)}")
                        all_responses.append(f"I couldn't process this part of your question: {q}")
                
                # Combine all responses
                combined_response = "\n\n".join([r for r in all_responses if r])
                return {
                    'success': True,
                    'formatted_response': combined_response,
                    'combined_from_multiple': True
                }
            else:
                # Single question, process normally
                return self._process_single_query(query)
                
        except Exception as e:
            logger.error(f"Error in process_analytics_query: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'formatted_response': (
                    "I'm having trouble processing your analytics request. "
                    "Please try rephrasing your question or ask about a specific metric."
                )
            }

    def _is_discount_aggregation_query(self, query: str) -> bool:
        """
        Detect if the query is about aggregating or ranking discounts per customer.
        """
        print("\n[DEBUG] ===== ENTERED _is_discount_aggregation_query =====")
        print(f"[DEBUG] Query: {query}")
        logger.info(f"[DEBUG] _is_discount_aggregation_query called with: {query}")
        
        # Call the helper and store the result
        result = self._is_discount_aggregation_query_helper(query)
        
        print(f"[DEBUG] _is_discount_aggregation_query result: {result}")
        logger.info(f"[DEBUG] _is_discount_aggregation_query result: {result}")
        return result

    def _is_discount_aggregation_query_helper(self, query: str) -> bool:
        q = query.lower()
        discount_keywords = ["discount", "discounts", "percent", "percentage", "off"]
        customer_keywords = ["customer", "customers", "buyer", "buyers", "person", "people"]
        ranking_keywords = ["most", "top", "highest", "biggest", "largest", "maximum", "max", "rank", "order", "sort"]
        if any(dk in q for dk in discount_keywords) and any(ck in q for ck in customer_keywords) and any(rk in q for rk in ranking_keywords):
            logger.info("[DEBUG] Discount aggregation pattern matched (keywords)")
            return True
        # Also match queries like "which customer got the most discount"
        if re.search(r"which.*customer.*(discount|discounts|percent|percentage|off)", q):
            logger.info("[DEBUG] Discount aggregation pattern matched (regex)")
            return True
        return False

    def _process_discount_aggregation_query(self, query: str) -> Dict[str, Any]:
        """
        Handle queries about which customers received the most/least discounts by aggregating discount percentages per customer.
        """
        try:
            logger.info("Processing discount aggregation query")
            
            # Formulate a more specific SQL query directly
            sql_query = """
            SELECT 
                c."Name" as customer_name,
                CAST(REPLACE(REPLACE(s."DiscountApplied", '%', ''), ' off', '') AS FLOAT) as discount_percentage,
                COUNT(*) as transaction_count
            FROM "sales_transactions" s
            JOIN "customer_info" c ON s."CustomerID" = c."CustomerID"
            WHERE s."DiscountApplied" IS NOT NULL AND s."DiscountApplied" != ''
            GROUP BY c."Name", discount_percentage
            ORDER BY discount_percentage ASC, transaction_count DESC
            """
            
            # Log the query we're about to execute
            logger.info(f"Executing direct SQL query for discount aggregation:")
            logger.info(sql_query)
            
            # Execute the query directly using the database connection
            from data import execute_query
            data = execute_query(sql_query)
            
            # Log the raw result for debugging
            logger.info(f"Query executed. Rows returned: {len(data) if data else 0}")
            if data:
                logger.info(f"Sample row: {data[0]}")
            
            if not data:
                # Try a more permissive query if no results
                logger.info("No results from first query, trying fallback query...")
                fallback_query = """
                SELECT 
                    c."Name" as customer_name,
                    s."DiscountApplied" as discount_percentage,
                    COUNT(*) as transaction_count
                FROM "sales_transactions" s
                JOIN "customer_info" c ON s."CustomerID" = c."CustomerID"
                WHERE s."DiscountApplied" IS NOT NULL
                GROUP BY c."Name", s."DiscountApplied"
                ORDER BY s."DiscountApplied" ASC, transaction_count DESC
                """
                data = execute_query(fallback_query)
                logger.info(f"Fallback query returned {len(data) if data else 0} rows")
            
            if not data:
                logger.warning("No customer discount data found in the database")
                return {
                    'success': True,
                    'formatted_response': "I couldn't find any customer discount data in the system to answer your question.",
                    'raw_results': []
                }
                
            # Process the data
            try:
                # Extract and format customer discounts
                customer_discounts = []
                
                for row in data:
                    try:
                        # Get customer name from the row
                        name = row.get('customer_name', 'Unknown Customer')
                        
                        # Get discount percentage, converting from string if necessary
                        discount = row.get('discount_percentage')
                        if isinstance(discount, str):
                            try:
                                # Remove any percentage signs or text
                                discount = float(''.join(c for c in discount if c.isdigit() or c == '.'))
                            except (ValueError, TypeError):
                                logger.warning(f"Could not parse discount value: {discount}")
                                continue
                        
                        # Only add if we have a valid discount
                        if discount is not None:
                            customer_discounts.append((name, float(discount)))
                    except Exception as e:
                        logger.warning(f"Error processing row {row}: {str(e)}")
                        continue
                
                if not customer_discounts:
                    logger.warning("No valid discount data found in the results")
                    return {
                        'success': True,
                        'formatted_response': "I found customer data, but couldn't extract any valid discount percentages.",
                        'raw_results': data
                    }
                
                # Sort by discount (ascending for least, descending for most)
                is_least = any(word in query.lower() for word in ['least', 'lowest', 'minimum'])
                customer_discounts.sort(key=lambda x: x[1], reverse=not is_least)
                
                # Get unique customers with their minimum discount
                seen_customers = set()
                unique_discounts = []
                for name, discount in customer_discounts:
                    if name not in seen_customers:
                        seen_customers.add(name)
                        unique_discounts.append((name, discount))
                
                # Build response
                lines = [f"Customers with the {'lowest' if is_least else 'highest'} discount percentages:"]
                for idx, (name, discount) in enumerate(unique_discounts[:10], 1):  # Show top 10
                    lines.append(f"{idx}. {name}: {discount}%")
                
                # If we have customers with 0% discount, mention them specifically
                zero_discount_customers = [name for name, disc in unique_discounts if disc == 0]
                if zero_discount_customers:
                    lines.append("\nNote: The following customers received no discount (0%):")
                    lines.extend(f"- {name}" for name in zero_discount_customers[:5])
                    if len(zero_discount_customers) > 5:
                        lines.append(f"... and {len(zero_discount_customers) - 5} more")
                
                return {
                    'success': True,
                    'formatted_response': "\n".join(lines),
                    'raw_results': data
                }
                
            except Exception as proc_error:
                logger.error(f"Error processing discount data: {str(proc_error)}", exc_info=True)
                return {
                    'success': False,
                    'formatted_response': f"I found some data but encountered an error while processing the discount information: {str(proc_error)}",
                    'raw_results': data
                }
        except Exception as e:
            logger.error(f"Error in _process_discount_aggregation_query: {str(e)}", exc_info=True)
            return {
                'success': False,
                'formatted_response': f"An error occurred while processing your discount aggregation query: {str(e)}",
                'raw_results': []
            }
    
    def _process_single_query(self, query: str) -> Dict[str, Any]:
        """Process a single analytics query."""
        try:
            # Extract parameters from the query
            params = self._extract_parameters(query)
            logger.info(f"Extracted parameters: {params}")
            
            # Generate natural language query
            nlq = self._generate_nlq(query, params)
            logger.info(f"Generated NLQ: {nlq}")
            
            # Execute the query and get results
            results = self._execute_queries(nlq, params)
            
            if not results or not any(results):
                logger.warning("No results returned from queries")
                return self._handle_no_data_response(query, params)
            
            # Generate insights from the results
            insights = self._generate_insights(query, results, params)
            
            # Format the response
            formatted_response = self._format_response(query, insights, results, params)
            
            return {
                'success': True,
                'formatted_response': formatted_response,
                'insights': insights,
                'raw_results': results
            }
            
        except json.JSONDecodeError as je:
            logger.error(f"JSON decode error: {str(je)}")
            return self._handle_no_data_response(query, params, "I'm having trouble understanding the data format.")
            
        except Exception as e:
            logger.error(f"Error processing analytics query: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'formatted_response': (
                    "I'm having trouble processing your analytics request. "
                    f"Error: {str(e)}. Please try rephrasing your question or ask about a different topic."
                )
            }
    
    def _handle_no_data_response(self, query: str, params: Dict[str, Any], additional_info: str = "") -> Dict[str, Any]:
        """Generate a helpful response when no data is available."""
        time_period = params.get('time_period', 'the selected time period')
        
        response = (
            f"I noticed you're asking about your shop's performance in {time_period}. "
            "I'm having trouble retrieving the data right now. This could be because:\n\n"
            "1. The data for this period hasn't been imported yet\n"
            "2. There might be an issue with the database connection\n"
            f"3. No sales data exists for {time_period}\n\n"
        )
        
        if additional_info:
            response += f"Additional information: {additional_info}\n\n"
            
        response += (
            "To help you better, could you please:\n"
            "1. Verify that the data for this period has been imported\n"
            "2. Try a different time period\n"
            "3. Ask a more specific question about your business metrics"
        )
        
        return {
            'success': False,
            'formatted_response': response,
            'error': 'No data available for the requested period'
        }
        
    def _extract_parameters(self, query: str) -> Dict[str, Any]:
        """
        Extract parameters from an analytics query.
        
        Args:
            query (str): The user's query
            
        Returns:
            Dict[str, Any]: Extracted parameters including time period, metrics, dimensions, etc.
        """
        # Enhanced parameter extraction
        params = {
            'time_period': None,  # e.g., 'last month', 'Q2 2023', 'last 30 days'
            'metrics': [],         # e.g., 'sales', 'revenue', 'conversion rate'
            'dimensions': [],      # e.g., 'by product', 'by region', 'by customer segment'
            'filters': {},         # e.g., 'for premium customers', 'in New York'
            'comparison': None     # e.g., 'vs last month', 'vs same period last year'
        }
        
        # Simple time period extraction (can be enhanced with more sophisticated NLP)
        time_phrases = ['last month', 'last week', 'last year', 'this month', 'this year', 
                       'last quarter', 'yesterday', 'today', 'last 7 days', 'last 30 days',
                       'last 90 days', 'last 12 months']
        
        for phrase in time_phrases:
            if phrase in query.lower():
                params['time_period'] = phrase
                break
                
        # Extract metrics (simplified)
        metric_keywords = {
            'sale': 'sales',
            'revenue': 'revenue',
            'profit': 'profit',
            'customer': 'customers',
            'growth': 'growth',
            'conversion': 'conversion_rate',
            'average order': 'average_order_value',
            'aov': 'average_order_value',
            'traffic': 'website_traffic'
        }
        
        for keyword, metric in metric_keywords.items():
            if keyword in query.lower() and metric not in params['metrics']:
                params['metrics'].append(metric)
                
        # If no specific metrics mentioned, use defaults
        if not params['metrics']:
            params['metrics'] = ['sales', 'revenue']
            
        return params
        
    def _split_into_questions(self, query: str) -> List[str]:
        """
        Split a query into individual questions if it contains multiple questions.
        
        Args:
            query (str): The input query that might contain multiple questions
            
        Returns:
            List[str]: List of individual questions
        """
        # Common question separators
        separators = ['?', '!', ';', '\n', '\t']
        
        # First, try to split on question marks
        questions = []
        current = []
        
        # Split on question marks but keep them
        parts = query.split('?')
        for i, part in enumerate(parts):
            if i < len(parts) - 1:
                # Add the question mark back to all but the last part
                part = part + '?'
            if part.strip():
                current.append(part.strip())
        
        # Further split on other separators
        final_questions = []
        for q in current:
            # Check if the question contains conjunctions that might separate multiple questions
            conjunctions = [' and ', ' or ', ' then ', ' also ', ' how about ']
            split_further = False
            
            for conj in conjunctions:
                if conj in q.lower():
                    # Split on the conjunction and add to final questions
                    sub_questions = [s.strip() for s in q.split(conj) if s.strip()]
                    final_questions.extend(sub_questions)
                    split_further = True
                    break
            
            if not split_further:
                final_questions.append(q)
        
        # Filter out very short questions (likely not actual questions)
        final_questions = [q for q in final_questions if len(q.split()) > 2]
        
        # If we somehow ended up with no questions, return the original as a single question
        if not final_questions:
            return [query]
            
        return final_questions
    
    def _generate_nlq(self, query: str, params: Dict[str, Any]) -> str:
        """
        Generate a structured query based on the extracted parameters.
        
        Args:
            query (str): The original user query
            params (Dict[str, Any]): Extracted parameters
            
        Returns:
            str: A structured query in JSON format
        """
        try:
            # Create a structured query based on the parameters
            structured_query = {
                'intent': 'analyze_retail_performance',
                'time_period': params.get('time_period', 'recent'),
                'metrics': params.get('metrics', ['sales']),
                'filters': {}
            }
            
            # Add any specific filters
            if 'category' in params:
                structured_query['filters']['category'] = params['category']
            if 'product' in params:
                structured_query['filters']['product'] = params['product']
                
            # Convert to JSON string
            return json.dumps(structured_query, indent=2)
            
        except Exception as e:
            logger.error(f"Error generating structured query: {str(e)}")
            # Fallback to the original query if we can't structure it
            return query
        
    def _execute_queries(self, nlq: str, params: Dict[str, Any]) -> List[Dict]:
        """
        Execute the natural language query and return the results.
        
        Args:
            nlq (str): The natural language query (in JSON format)
            params (Dict[str, Any]): Extracted parameters
            
        Returns:
            List[Dict]: List of query results with metadata
        """
        try:
            # Try to parse the NLQ as JSON
            try:
                query_data = json.loads(nlq)
            except json.JSONDecodeError:
                # If it's not JSON, treat it as a simple string query
                query_data = {'query': nlq}
            
            logger.info(f"Executing query: {json.dumps(query_data, indent=2)}")
            
            # Execute the query using the NLQ module
            result = self.execute_nl_query(nlq)
            
            # Log the result (truncate if too large)
            result_str = json.dumps(result, cls=DecimalEncoder, indent=2)
            logger.debug(f"Query result: {result_str[:500]}...")
            
            if not result:
                logger.warning("Query returned no results")
                return []
                
            return [{
                'query': nlq,
                'data': result,
                'metadata': {
                    'execution_time': datetime.now().isoformat(),
                    'parameters': params
                }
            }]
            
        except json.JSONDecodeError as je:
            logger.error(f"JSON decode error in query execution: {str(je)}")
            return [{
                'query': nlq,
                'error': f"Data format error: {str(je)}",
                'metadata': {'error': True}
            }]
            
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}", exc_info=True)
            return [{
                'query': nlq,
                'error': str(e),
                'metadata': {'error': True}
            }]
    
    def _generate_insights(self, query: str, results: List[Dict], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate retail-focused insights from the query results.
        
        Args:
            query (str): The original user query
            results (List[Dict]): List of query results
            params (Dict[str, Any]): Extracted parameters
            
        Returns:
            Dict[str, Any]: Generated insights with retail focus
        """
        insights = {
            'key_metrics': [],
            'trends': [],
            'anomalies': [],
            'recommendations': []
        }
        
        # Process results to generate retail-specific insights
        if results and len(results) > 0 and 'data' in results[0] and results[0]['data']:
            data = results[0]['data']
            if isinstance(data, list) and len(data) > 0:
                # Add retail-specific metrics
                total_sales = sum(float(item.get('TotalAmount', 0)) for item in data if item.get('TotalAmount'))
                avg_transaction = total_sales / len(data) if data else 0
                
                insights['key_metrics'].extend([
                    {
                        'metric': 'Total Sales',
                        'value': f'₹{total_sales:,.2f}',
                        'description': f'Total sales amount for the selected period: ₹{total_sales:,.2f}'
                    },
                    {
                        'metric': 'Average Transaction Value',
                        'value': f'₹{avg_transaction:,.2f}',
                        'description': f'Average amount per transaction: ₹{avg_transaction:,.2f}'
                    },
                    {
                        'metric': 'Transaction Count',
                        'value': len(data),
                        'description': f'Total number of transactions: {len(data)}'
                    }
                ])
                
                # Add retail-specific recommendations
                insights['recommendations'].extend([
                    {
                        'title': 'Inventory Optimization',
                        'description': 'Consider restocking best-selling items to avoid stockouts.',
                        'priority': 'high'
                    },
                    {
                        'title': 'Customer Engagement',
                        'description': 'Launch a loyalty program to increase customer retention.',
                        'priority': 'medium'
                    },
                    {
                        'title': 'Promotional Strategy',
                        'description': 'Create bundle offers for frequently purchased together items.',
                        'priority': 'medium'
                    }
                ])
                
                # Add trend analysis
                if len(data) > 1:
                    insights['trends'].append({
                        'name': 'Sales Trend',
                        'description': 'Analyzing daily sales patterns to identify peak hours and days.'
                    })
        
        return insights
    
    def _format_response(self, query: str, insights: Dict[str, Any], 
                         results: List[Dict], params: Dict[str, Any]) -> str:
        """
        Format the response to be retail-focused and user-friendly.
        
        Args:
            query (str): The original user query
            insights (Dict[str, Any]): Generated insights
            results (List[Dict]): Query results
            params (Dict[str, Any]): Extracted parameters
            
        Returns:
            str: Formatted response string with retail focus
        """
        response = f"🔍 I've analyzed your retail query: '{query}'\n\n"
        
        # Add key metrics
        if insights.get('key_metrics'):
            response += "📈 Retail Performance Summary:\n"
            for metric in insights['key_metrics']:
                response += f"• {metric['description']}\n"
            response += "\n"
        
        # Add trends if available
        if insights.get('trends'):
            response += "📊 Sales Trends:\n"
            for trend in insights['trends']:
                response += f"• {trend['description']}\n"
            response += "\n"
        
        # Add retail-specific recommendations
        if insights.get('recommendations'):
            response += "💡 Business Improvement Suggestions:\n"
            for rec in insights['recommendations']:
                priority_emoji = "🟢" if rec.get('priority') == 'low' else "🟡" if rec.get('priority') == 'medium' else "🔴"
                response += f"{priority_emoji} {rec['description']}\n"
            response += "\n"
        
        # Add a retail-focused call to action
        response += (
            "🛍️ As your retail assistant, I can help you with:\n"
            "• Analyzing sales performance by product/category\n"
            "• Identifying customer buying patterns\n"
            "• Optimizing inventory levels\n"
            "• Creating promotional strategies\n\n"
            "What would you like to explore further?"
        )
        
        return response
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze the query to determine if it's an analytics query and extract relevant parameters.
        
        Args:
            query (str): The user's query
            
        Returns:
            Dict[str, Any]: Analysis result with 'is_analytics_query' flag and extracted parameters
        """
        try:
            # Check for analytics-related keywords
            analytics_keywords = [
                'analyze', 'analysis', 'trend', 'insight', 'recommend', 'suggest',
                'improve', 'optimize', 'performance', 'efficiency', 'strategy',
                'opportunity', 'why', 'how to', 'what should', 'business case',
                'comparative analysis', 'benchmark', 'kpi', 'metric', 'growth',
                'forecast', 'prediction', 'pattern', 'correlation', 'impact'
            ]
            
            # Check if any analytics keyword is in the query
            is_analytics_query = any(keyword in query.lower() for keyword in analytics_keywords)
            
            # If no direct keyword match, use LLM to determine if it's an analytics query
            if not is_analytics_query:
                is_analytics_query = self._check_analytics_intent(query)
            
            return {
                'is_analytics_query': is_analytics_query,
                'confidence': 0.9 if is_analytics_query else 0.1,
                'query_type': 'analytics',
                'parameters': {}
            }
            
        except Exception as e:
            logger.error(f"Error in analyze_query: {str(e)}")
            return {'is_analytics_query': False, 'confidence': 0.0}
    
    def _check_analytics_intent(self, query: str) -> bool:
        """
        Use LLM to determine if the query has analytical intent.
        
        Args:
            query (str): The user's query
            
        Returns:
            bool: True if the query has analytical intent, False otherwise
        """
        try:
            system_prompt = """
            You are an intent classifier. Determine if the user's query requires analytical processing.
            A query requires analytical processing if it asks for:
            - Analysis of trends or patterns
            - Business recommendations or improvements
            - Insights from multiple data points
            - Why something is happening
            - What actions to take based on data
            - Customer behavior analysis
            - Purchase patterns or frequency
            - Identifying regular or loyal customers
            - Customer segmentation or categorization
            - Analysis of buying habits
            
            Examples of analytical queries:
            - "Which customers buy from me regularly?"
            - "What are the buying patterns of my top customers?"
            - "Show me customers who purchase every month"
            - "Who are my most loyal customers?"
            - "Analyze customer purchase frequency"
            
            Respond with only 'YES' or 'NO' in uppercase.
            """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=query)
            ]
            
            response = self.llm.invoke(messages)
            return response.content.strip().upper() == 'YES'
            
        except Exception as e:
            logger.error(f"Error in _check_analytics_intent: {str(e)}")
            return False
    
    def process_analytics_query(self, query: str) -> Dict[str, Any]:
        """
        Process an analytics query by breaking it down into sub-queries,
        gathering data, and generating insights.
        
        Args:
            query (str): The user's analytics query
            
        Returns:
            Dict[str, Any]: Response containing analysis and recommendations
        """
        try:
            # Step 1: Break down the analytics query into sub-queries
            sub_queries = self._generate_sub_queries(query)
            
            # Step 2: Get data for each sub-query
            query_results = {}
            for i, sub_query in enumerate(sub_queries, 1):
                try:
                    # Execute the sub-query using the NLQ module
                    logger.info(f"Executing sub-query {i}: {sub_query}")
                    result = self.execute_nl_query(sub_query)
                    
                    if result.get('success', False):
                        query_results[f"query_{i}"] = {
                            "question": sub_query,
                            "sql": result.get('sql', ''),
                            "data": result.get('data', []),
                            "success": True
                        }
                    else:
                        query_results[f"query_{i}"] = {
                            "question": sub_query,
                            "error": result.get('error', 'Unknown error'),
                            "success": False
                        }
                        logger.error(f"Error in sub-query {i}: {result.get('error')}")
                        
                except Exception as e:
                    error_msg = f"Error processing sub-query {i}: {str(e)}"
                    logger.error(error_msg)
                    query_results[f"query_{i}"] = {
                        "question": sub_query,
                        "error": error_msg,
                        "success": False
                    }
            
            # Step 3: Generate insights from the collected data
            insights = self._generate_insights(query, query_results)
            
            # Step 4: Generate recommendations
            recommendations = self._generate_recommendations(query, insights)
            
            return {
                "status": "success",
                "query": query,
                "sub_queries": query_results,
                "insights": insights,
                "recommendations": recommendations,
                "formatted_response": self._format_response(query, insights, recommendations)
            }
            
        except Exception as e:
            error_msg = f"Error processing analytics query: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "formatted_response": "I encountered an error while processing your analytics request. Please try again later."
            }
    
    def _generate_sub_queries(self, query: str) -> List[str]:
        """
        Break down an analytics query into smaller, answerable sub-queries.
        
        Args:
            query (str): The original analytics query
            
        Returns:
            List[str]: List of sub-queries
        """
        try:
            # Check if this is a discount-related query
            if any(term in query.lower() for term in ['discount', 'discounts', 'off', '%']):
                # For discount queries, use a direct SQL query
                is_least = any(word in query.lower() for word in ['least', 'lowest', 'minimum'])
                order = "ASC" if is_least else "DESC"
                
                # Direct SQL query to get customer discounts
                return [f"""
                    SELECT 
                        c."Name" as customer_name,
                        s."DiscountApplied" as discount_percentage,
                        COUNT(*) as transaction_count
                    FROM "sales_transactions" s
                    JOIN "customer_info" c ON s."CustomerID" = c."CustomerID"
                    WHERE s."DiscountApplied" IS NOT NULL 
                      AND s."DiscountApplied" <> ''
                      AND s."DiscountApplied" ~ '^[0-9]+$'  -- Only numeric values
                    GROUP BY c."Name", s."DiscountApplied"
                    ORDER BY CAST(s."DiscountApplied" AS INTEGER) {order}, transaction_count DESC
                    LIMIT 10
                """]
                
            # Special handling for other customer-related queries
            if any(word in query.lower() for word in ['customer', 'customers', 'buy', 'purchase', 'regularly', 'frequent']):
                # Use simpler queries that match the database schema
                return [
                    "Show me customer names and their total number of purchases",
                    "List customers and their most recent purchase date",
                    "Show customers who made more than 3 purchases"
                ]
                
            # For other queries, use LLM to generate sub-queries
            prompt = f"""
            Break down the following business analytics question into 2-5 specific, answerable sub-questions 
            that would help provide a comprehensive answer. Focus on identifying key metrics, time periods, 
            and comparison points. Return the sub-questions as a JSON array of strings.
            
            Question: {query}
            
            Respond with ONLY a valid JSON array. Example:
            ["What are the sales trends for the past 6 months?", "How does this compare to the same period last year?"]
            """
            
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            
            # Parse the response as JSON
            try:
                sub_queries = json.loads(response.content)
                if not isinstance(sub_queries, list):
                    raise ValueError("Response is not a list")
                return [str(q) for q in sub_queries]
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse sub-queries, using fallback: {str(e)}")
                # Fallback to direct query
                return [query]
                
        except Exception as e:
            logger.error(f"Error in _generate_sub_queries: {str(e)}")
            # Fallback to the original query
            return [query]
    
    def _generate_insights(self, original_query: str, query_results: Dict[str, Any]) -> List[str]:
        """
        Generate insights from the collected query results.
        
        Args:
            original_query (str): The original user query
            query_results (Dict[str, Any]): Results from sub-queries
            
        Returns:
            List[str]: List of insights
        """
        try:
            if not query_results:
                return ["No data available to generate insights."]
                
            insights = []
            
            # Check if we have any successful queries
            successful_queries = [q for q in query_results.values() if q.get('success', False)]
            if not successful_queries:
                return ["Unable to generate insights: no successful queries."]
            
            # Analyze the first successful query result
            first_result = successful_queries[0]
            data = first_result.get('data', [])
            
            if not data:
                return ["No data found for the given query."]
                
            # Basic analysis based on the data
            if any(word in original_query.lower() for word in ['customer', 'customers', 'buy', 'purchase', 'regularly', 'frequent']):
                if len(data) > 0:
                    # Get column names from the first row of data
                    columns = list(data[0].keys()) if data else []
                    
                    # Handle different possible column names
                    name_columns = [col for col in ['customer_name', 'name', 'customer', 'customername'] if col in columns]
                    count_columns = [col for col in ['purchase_count', 'count', 'total_purchases', 'transactions'] if col in columns]
                    date_columns = [col for col in ['last_purchase', 'recent_purchase', 'purchase_date', 'date'] if col in columns]
                    
                    # Extract customer names if available
                    if name_columns:
                        name_col = name_columns[0]
                        customer_names = [str(row.get(name_col, 'Unknown')).strip() for row in data if row.get(name_col)]
                        if customer_names:
                            top_customers = ", ".join(customer_names[:3])  # Show top 3
                            if len(customer_names) > 3:
                                insights.append(f"Your top customers include: {top_customers}, and {len(customer_names) - 3} others.")
                            else:
                                insights.append(f"Your customers are: {top_customers}.")
                    
                    # Analyze purchase counts if available
                    if count_columns:
                        count_col = count_columns[0]
                        try:
                            purchase_counts = [int(row[count_col]) for row in data if row.get(count_col) is not None]
                            if purchase_counts:
                                total_purchases = sum(purchase_counts)
                                avg_purchases = total_purchases / len(purchase_counts)
                                max_purchases = max(purchase_counts)
                                
                                insights.append(f"Total purchases across all customers: {total_purchases}")
                                insights.append(f"Average purchases per customer: {avg_purchases:.1f}")
                                
                                # Find top customer by purchase count
                                if name_columns:
                                    top_customer = max(data, key=lambda x: x.get(count_col, 0))
                                    insights.append(f"Top customer by purchases: {top_customer.get(name_columns[0], 'Unknown')} with {top_customer.get(count_col)} purchases")
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Could not process purchase counts: {str(e)}")
                    
                    # Analyze purchase dates if available
                    if date_columns:
                        date_col = date_columns[0]
                        try:
                            dates = [row[date_col] for row in data if row.get(date_col)]
                            if dates:
                                # Convert string dates to datetime objects if needed
                                if isinstance(dates[0], str):
                                    dates = [datetime.strptime(d, '%Y-%m-%d') for d in dates if d]
                                
                                if dates:
                                    most_recent = max(dates)
                                    oldest = min(dates)
                                    date_range = (most_recent - oldest).days
                                    
                                    insights.append(f"Purchase history spans {date_range} days (from {oldest.strftime('%b %d, %Y')} to {most_recent.strftime('%b %d, %Y')})")
                                    
                                    # Calculate recency in days
                                    today = datetime.now()
                                    days_since_last = (today - most_recent).days
                                    if days_since_last == 0:
                                        insights.append("Most recent purchase was today!")
                                    else:
                                        insights.append(f"Most recent purchase was {days_since_last} days ago")
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Could not process purchase dates: {str(e)}")
            
            # Add more specific analysis based on the query type
            if 'sales' in original_query.lower() or 'purchase' in original_query.lower():
                total_sales = sum(row.get('total_amount', 0) for row in data if 'total_amount' in row)
                if total_sales > 0:
                    insights.append(f"Total sales amount: ₹{total_sales:,.2f}")
                
                avg_sale = total_sales / len(data) if data else 0
                if avg_sale > 0:
                    insights.append(f"Average transaction value: ₹{avg_sale:,.2f}")
            
            # If no specific insights were generated, provide a generic response
            if not insights:
                insights.append(f"Found {len(data)} records matching your query.")
                
            return insights
            
        except Exception as e:
            logger.error(f"Error in _generate_insights: {str(e)}")
            if 'data' in locals() and data:
                return ["I found some information, but had trouble analyzing it. Here are the raw results:", 
                       str([{k: v for k, v in row.items() if not k.startswith('_')} for row in data][:3])]
            return ["Unable to generate specific insights due to data limitations."]
    
    def _generate_recommendations(self, original_query: str, insights: List[str]) -> List[str]:
        """
        Generate actionable recommendations based on insights.
        
        Args:
            original_query (str): The original user query
            insights (List[str]): Generated insights
            
        Returns:
            List[str]: List of recommendations
        """
        try:
            # In a real implementation, this would analyze the insights
            # For now, we'll generate some placeholder recommendations
            return [
                "Consider increasing marketing spend on the Electronics category to capitalize on its popularity.",
                "Implement a customer loyalty program to further improve retention rates.",
                "Analyze the reasons behind the sales growth to replicate success in other areas."
            ]
        except Exception as e:
            logger.error(f"Error in _generate_recommendations: {str(e)}")
            return ["No specific recommendations available at this time."]
    
    def _format_response(self, query: str, insights: List[str], recommendations: List[str]) -> str:
        """
        Format the analytics response in a user-friendly way.
        
        Args:
            query (str): The original user query
            insights (List[str]): Generated insights
            recommendations (List[str]): Generated recommendations
            
        Returns:
            str: Formatted response
        """
        try:
            response = ["🔍 Analysis Results 🔍", ""]
            
            # Add a summary of the query
            response.append(f"📋 You asked: {query}")
            
            if insights:
                response.append("\n📊 Here's what I found:")
                for insight in insights:
                    # Skip empty insights
                    if insight and insight.strip():
                        response.append(f"• {insight.strip()}")
            
            if recommendations:
                response.append("\n💡 Recommendations:")
                for rec in recommendations:
                    # Skip empty recommendations
                    if rec and rec.strip():
                        response.append(f"• {rec.strip()}")
            
            # If no insights or recommendations, provide a helpful message
            if not any(insights) and not any(recommendations):
                response.append("\nI couldn't find any specific data to answer your query. "
                             "This might be because we don't have enough data yet or the query needs to be more specific.")
            
            # Add a friendly sign-off
            response.append("\nIs there anything else you'd like to know? 😊")
            
            # Join all parts with newlines and ensure no double newlines
            formatted = "\n".join(part for part in response if part)
            return formatted.replace("\n\n\n", "\n\n").strip()
            
        except Exception as e:
            logger.error(f"Error in _format_response: {str(e)}")
            return "I encountered an error while formatting the response. Please try again later."

# Example usage
if __name__ == "__main__":
    agent = AnalyticsAgent()
    
    test_queries = [
        "How can I improve my business?",
        "What are the key trends in our sales data?",
        "Why are our customer retention rates changing?",
        "What's the weather like today?"  # Should not be an analytics query
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"QUERY: {query}")
        analysis = agent.analyze_query(query)
        print(f"Analysis: {analysis}")
        
        if analysis.get('is_analytics_query', False):
            print("\nProcessing as analytics query...")
            result = agent.process_analytics_query(query)
            print(f"\n{result['formatted_response']}")
