import json
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum, auto
from dataclasses import dataclass, field

# Import the formatting function from response_utils
from response_utils import format_database_response

# Import the execute_nl_query function from nlq.py
from nlq import execute_nl_query

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class QueryStatus(Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class QueryStep:
    """Represents a single query in the analytics pipeline."""
    query_id: str
    sql: str
    purpose: str
    status: QueryStatus = QueryStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

@dataclass
class AnalyticsSession:
    """Tracks the state of an analytics session."""
    session_id: str
    user_question: str
    steps: List[QueryStep] = field(default_factory=list)
    current_step: int = 0
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def add_step(self, sql: str, purpose: str) -> str:
        """Add a new query step to the session."""
        query_id = f"q{len(self.steps) + 1}"
        step = QueryStep(
            query_id=query_id,
            sql=sql,
            purpose=purpose
        )
        self.steps.append(step)
        self.updated_at = datetime.utcnow()
        return query_id

    def update_step_status(self, query_id: str, status: QueryStatus, result: Optional[Dict] = None, error: Optional[str] = None):
        """Update the status of a query step."""
        for step in self.steps:
            if step.query_id == query_id:
                step.status = status
                step.completed_at = datetime.utcnow()
                if result is not None:
                    step.result = result
                if error is not None:
                    step.error = error
                self.updated_at = datetime.utcnow()
                break

# Helper to assess result quality
def is_insufficient(result: Dict[str, Any], min_rows: int = 5, max_variance: float = 0.1) -> bool:
    data = result.get('data', []) or []
    # Too few rows
    if len(data) < min_rows:
        return True
    # Check variance in a numeric column if present
    if data and 'total_spent' in data[0]:
        values = [r.get('total_spent', 0) for r in data]
        if max(values) - min(values) < max_variance * (sum(values) / len(values)):
            return True
    return False

class AnalyticsAgent:
    """Agent for handling complex analytics queries by breaking them down into multiple SQL queries."""
    
    def __init__(self, session_id: str):
        self.session = AnalyticsSession(session_id=session_id, user_question="")
        self.llm = None
        
    def analyze_question(self, question: str) -> Dict[str, Any]:
        """Analyze the question and plan the initial query."""
        self.session.user_question = question
        self.session.steps = []

        # Seed with a general top-spenders query
        seed_prompt = f"""
        You are an analytics assistant.
        User goal: {question}
        Produce one SQL query that would give you actionable insights to improve the business. Return only the SQL statement.
        """
        prompt_result = execute_nl_query(seed_prompt)
        sql = prompt_result.get('sql', '').strip()
        purpose = "Initial analysis: key customer spenders"
        self.session.add_step(sql=sql, purpose=purpose)

        logger.info(f"[Analytics Agent] Planned {len(self.session.steps)} steps for question: {question}")
        return {"status": "success", "steps_planned": len(self.session.steps), "next_step": 0}
    
    def execute_next_step(self) -> Tuple[bool, Dict[str, Any]]:
        """Execute the next planned query step, assess, and optionally queue a follow-up."""
        if self.session.current_step >= len(self.session.steps):
            return False, {"status": "complete", "message": "All steps completed"}
        step = self.session.steps[self.session.current_step]
        logger.info(f"[Analytics Agent] Executing {step.query_id}: {step.purpose}")
        logger.debug(f"[Analytics Agent] SQL: {step.sql}")
        try:
            result = execute_nl_query(step.sql)
            success = result.get('success', False)
            step.status = QueryStatus.COMPLETED
            step.result = result
            step.completed_at = datetime.utcnow()

            # Assess result
            if success and is_insufficient(result):
                logger.info(f"[Analytics Agent] Result insufficient for {step.query_id}, generating follow-up")
                sample = result.get('data', [])[:5]
                followup_prompt = (
                    f"User goal: {self.session.user_question}\n"
                    f"Here is a sample of the result: {json.dumps(sample, default=str)}\n"
                    "What SQL query should run next to get more actionable insight? Return only the SQL."
                )
                follow_sql = execute_nl_query(followup_prompt).get('sql', '').strip()
                self.session.add_step(sql=follow_sql, purpose="Follow-up analysis: drill deeper")
            # Advance to next
            self.session.current_step += 1
            has_next = self.session.current_step < len(self.session.steps)
            return has_next, {"status": "success", "step_executed": step.query_id}
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[Analytics Agent] Error: {error_msg}\n{traceback.format_exc()}")
            step.status = QueryStatus.FAILED
            step.error = error_msg
            return False, {"status": "error", "error": error_msg, "step": step.query_id}

    def get_final_response(self) -> Dict[str, Any]:
        """Generate a final narrative based on all results."""
        completed_steps = [s for s in self.session.steps if s.status == QueryStatus.COMPLETED]
        data_summary = []
        for s in completed_steps:
            data_summary.append({
                'query_id': s.query_id,
                'purpose': s.purpose,
                'sample': s.result.get('data', [])[:3]
            })
        # Use LLM to craft narrative
        narrative_prompt = (
            f"Based on these analyses for '{self.session.user_question}', summarize key findings and provide 3 actionable recommendations.\n"
            f"Analyses summary: {json.dumps(data_summary, default=str)}"
        )
        llm_resp = None
        try:
            llm_resp = execute_nl_query(narrative_prompt)
        except Exception:
            pass
        answer = llm_resp.get('answer') if llm_resp else "I have insights ready for you."
        return {
            "session_id": self.session.session_id,
            "question": self.session.user_question,
            "status": "completed",
            "steps_completed": len(completed_steps),
            "data_summary": data_summary,
            "answer": answer
        }

# Node function remains unchanged

def analytics_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    if state.get("processed", False):
        return {"agent_output": {"message": "Query already processed", "success": False}}
    query = state.get("query", "")
    session_id = state.get("session_id", f"sess_{int(datetime.utcnow().timestamp())}")
    agent = AnalyticsAgent(session_id=session_id)
    agent.analyze_question(query)
    while True:
        has_next, _ = agent.execute_next_step()
        if not has_next:
            break
    response = agent.get_final_response()
    return {"agent_output": {"success": True, "response": response, "data": response.get("data_summary"), "session_id": session_id}}
