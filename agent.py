import os
import json
from typing import Dict, TypedDict, Annotated, List, Literal, Union, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from nlq import generate_sql_query, execute_nl_query
import decimal

# Custom JSON encoder to handle Decimal types
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            return float(obj)  # Convert Decimal to float for JSON serialization
        return super().default(obj)

# Load environment variables
load_dotenv()

# Initialize Gemini 1.5 Flash model for the supervisor
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Define the state schema
class AgentState(TypedDict):
    messages: Annotated[List[Union[HumanMessage, AIMessage]], lambda x, y: x + y]
    next: Literal["supervisor", "db_query_agent", END]
    query: str
    selected_agent: str
    agent_output: str
    final_answer: str
    processed: bool

# Define the available agents
agents = {
    "db_query_agent": {
        "description": "Useful for querying the database to get specific data points or analytics."
    },
    # Future agents can be added here
}

# Define the supervisor node
def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """
    The supervisor node that decides which agent to use based on the user's query.
    Uses an LLM to make the routing decision.
    """
    # Get the query from state or use the last message
    query = state.get("query")
    if not query and state["messages"]:
        query = state["messages"][-1].content if hasattr(state["messages"][-1], 'content') else str(state["messages"][-1])
    
    if not query:
        return {"next": "db_query_agent", "selected_agent": "db_query_agent"}
    
    # Create a structured prompt for the supervisor
    prompt = f"""You are a supervisor agent responsible for routing user queries to the most appropriate agent.
    
    Available agents:
    - db_query_agent: For querying database to get specific data points or analytics.
    
    User query: "{query}"
    
    Your task is to analyze the user's query and determine which agent should handle it.
    
    Respond with a JSON object containing the agent name, like this:
    {{
        "agent": "db_query_agent",
        "reasoning": "Brief explanation of why this agent was chosen"
    }}
    """
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        result = json.loads(response.content)
        if isinstance(result, dict) and "agent" in result:
            selected_agent = result["agent"]
        else:
            selected_agent = "db_query_agent"
    except Exception:
        selected_agent = "db_query_agent"
    
    # Ensure the selected agent is valid
    if selected_agent not in agents:
        selected_agent = "db_query_agent"
    
    return {
        "next": selected_agent,
        "selected_agent": selected_agent,
        "query": query
    }

# Define the db_query_agent node
def db_query_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    The database query agent that handles natural language to SQL conversion and execution.
    """
    if state.get("processed", False):
        return {"agent_output": "Query already processed"}
    
    query = state.get("query", "")
    print(f"\n[DB Query Agent] Processing query: {query}")
    
    try:
        # Execute the query and get detailed results
        result = execute_nl_query(query)
        
        # Print debug information
        print("\n=== Query Execution Details ===")
        print(f"Original Query: {query}")
        if "sql" in result:
            print(f"Generated SQL: {result['sql']}")
        if "assumptions" in result and result["assumptions"]:
            print("\nAssumptions:")
            for i, assumption in enumerate(result["assumptions"], 1):
                print(f"{i}. {assumption}")
        if "notes" in result and result["notes"]:
            print(f"\nNotes: {result['notes']}")
        if "data" in result and result["data"]:
            print("\nResult Data:")
            print(json.dumps(result["data"], indent=2, cls=DecimalEncoder))
        print("=" * 40 + "\n")
        
        if not result.get("success", False):
            error_msg = result.get("error", "Unknown error")
            print(f"[DB Query Agent] Error: {error_msg}")
            return {
                "agent_output": result,
                "status": "error"
            }
        
        return {
            "agent_output": json.dumps(result, cls=DecimalEncoder),
            "status": "success"
        }
                
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[DB Query Agent] Unexpected error: {error_details}")
        return {
            "agent_output": f"Error processing query: {str(e)}\n{error_details}",
            "status": "error"
        }

# Define the final response node
def final_response_node(state: AgentState) -> Dict[str, str]:
    """
    Format the final response to the user based on the agent's output.
    """
    try:
        agent_output = json.loads(state["agent_output"])
        query = state["query"]
        # Compose a prompt for the LLM to summarize the result
        data = agent_output.get("data") or agent_output.get("results") or agent_output
        prompt = f"""
You are a helpful assistant. Given the following user query and the output data from a database, generate a concise, natural language answer for the user. If the data is empty or no results are found, say so politely. Otherwise, summarize the key findings in a user-friendly way.

User Query: {query}

Database Output:
{json.dumps(data, indent=2, ensure_ascii=False, cls=DecimalEncoder)}

Answer: """
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = response.content.strip()
        return {"final_answer": answer}
    except Exception as e:
        return {"final_answer": f"An error occurred while processing your request: {str(e)}"}

# Create the workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("db_query_agent", db_query_agent_node)
workflow.add_node("final_response", final_response_node)

# Define the edges
workflow.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {
        "db_query_agent": "db_query_agent",
        END: END,
    },
)

workflow.add_edge("db_query_agent", "final_response")
workflow.add_edge("final_response", END)

# Set the entry point
workflow.set_entry_point("supervisor")

# Compile the workflow
runnable = workflow.compile()

def process_query(query: str) -> str:
    """
    Process a natural language query through the agent system.
    
    Args:
        query (str): The natural language query from the user
        
    Returns:
        str: The final formatted response
    """
    print(f"\n{'='*50}")
    print(f"Processing query: {query}")
    print(f"{'='*50}")
    
    # Initialize the state
    state = {
        "messages": [HumanMessage(content=query)],
        "next": "supervisor",
        "query": "",
        "selected_agent": "",
        "agent_output": "",
        "final_answer": "",
        "processed": False
    }
    
    # Run the workflow
    final_state = runnable.invoke(state)
    
    # Print the final answer if available
    if final_state.get("final_answer"):
        print(f"\n✅ Successfully processed query: {query}")
        print(f"📊 Result: {final_state['final_answer']}")
        return final_state['final_answer']
    else:
        print("\n❌ Failed to process query. No final answer was generated.")
        if final_state.get("agent_output"):
            print(f"Agent output: {final_state['agent_output']}")
        return "I'm sorry, I couldn't process your request."

if __name__ == "__main__":
    import sys
    
    # Get query from command line or use default
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is my income on 14th April?"
    
    print("\n" + "="*50)
    print(f"Processing query: {query}")
    print("="*50 + "\n")
    
    # Initialize the state
    state = {
        "query": query,
        "messages": [HumanMessage(content=query)],
        "processed": False
    }
    
    try:
        # Run the workflow
        final_state = runnable.invoke(state)
        
        # Print the final answer if available
        if final_state.get("final_answer"):
            print(f"\n✅ Successfully processed query: {query}")
            print(f"📊 Result: {final_state['final_answer']}")
        else:
            print("\n❌ Failed to process query. No final answer was generated.")
            if final_state.get("agent_output"):
                print(f"Agent output: {final_state['agent_output']}")
    
    except Exception as e:
        print(f"\n❌ An error occurred while processing the query: {str(e)}")
