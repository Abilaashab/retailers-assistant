import json
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum, auto
from dataclasses import dataclass, field

# Import the formatting function from response_utils
from response_utils import format_database_response
# Import execute_nl_query to generate SQL and narrative
from nlq import execute_nl_query, summarize_metrics

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
        query_id = f"q{len(self.steps) + 1}"
        step = QueryStep(query_id=query_id, sql=sql, purpose=purpose)
        self.steps.append(step)
        self.updated_at = datetime.utcnow()
        # Initialize result cache if not present
        if 'result' not in self.context:
            self.context['result'] = None
        return query_id

    def update_step_status(self, query_id: str, status: QueryStatus,
                           result: Optional[Dict] = None, error: Optional[str] = None):
        for step in self.steps:
            if step.query_id == query_id:
                step.status = status
                step.completed_at = datetime.utcnow()
                if result is not None:
                    # Store result in session context to avoid duplicate execution
                    self.context['result'] = result
                    step.result = result
                if error is not None:
                    step.error = error
                self.updated_at = datetime.utcnow()
                break

# Helper to assess result quality
def is_insufficient(result: Dict[str, Any], min_rows: int = 5,
                   max_variance: float = 0.1) -> bool:
    data = result.get('data', []) or []
    if len(data) < min_rows:
        return True
    if data and 'total_spent' in data[0]:
        values = [r.get('total_spent', 0) for r in data]
        avg = sum(values) / len(values)
        if max(values) - min(values) < max_variance * avg:
            return True
    return False

class AnalyticsAgent:
    """Agent for handling complex analytics queries by breaking them down into multiple SQL queries."""
    MAX_STEPS = 5
    _query_cache = {}  # Cache for SQL query results
    _prompt_cache = {}  # Cache for prompt-based generations
    _sql_result_cache = {}  # Cache for SQL execution results

    def __init__(self, session_id: str):
        self.session = AnalyticsSession(session_id=session_id, user_question="")
        self._cache_key = None

    def _get_from_cache(self, key: str, cache_type: str = 'query') -> Optional[Any]:
        """Retrieve a value from the specified cache."""
        cache = {
            'prompt': self._prompt_cache,
            'result': self._sql_result_cache,
            'query': self._query_cache
        }.get(cache_type, self._query_cache)
        return cache.get(key)

    def _add_to_cache(self, key: str, value: Any, cache_type: str = 'query') -> None:
        """Add a value to the specified cache."""
        cache = {
            'prompt': self._prompt_cache,
            'result': self._sql_result_cache,
            'query': self._query_cache
        }.get(cache_type, self._query_cache)
        cache[key] = value
        if cache_type == 'query':
            self._cache_key = key

    def _get_sql_from_prompt(self, prompt: str, purpose: str) -> str:
        """Get SQL from prompt, using cache if available."""
        prompt_hash = f"prompt_{hash(prompt)}"
        
        # Try to get from prompt cache first
        cached_sql = self._get_from_cache(prompt_hash, 'prompt')
        if cached_sql:
            logger.debug(f"Using cached SQL for prompt: {purpose}")
            return cached_sql
            
        # Generate new SQL if not in cache
        logger.debug(f"Generating new SQL for: {purpose}")
        gen = execute_nl_query(prompt)
        sql = gen.get('sql', '').strip()
        
        # Cache the result
        if sql:
            self._add_to_cache(prompt_hash, sql, 'prompt')
            self._add_to_cache(f"sql_{hash(sql)}", sql)  # Also cache by SQL content
            
        return sql

    def analyze_question(self, question: str) -> Dict[str, Any]:
        self.session.user_question = question
        self.session.steps = []
        
        # Generate initial SQL using prompt cache
        seed_prompt = (
            f"You are an analytics assistant.\n"
            f"User goal: {question}\n"
            "Produce one SQL query that gives actionable insights. Return only the SQL."
        )
        
        sql = self._get_sql_from_prompt(seed_prompt, "initial analysis")
        if not sql:
            return {"status": "error", "error": "Failed to generate SQL query"}
            
        self.session.add_step(sql=sql, purpose="Initial analysis")
        logger.info(f"Planned 1 step for question: {question}")
        return {"status": "success", "steps_planned": 1, "next_step": 0}

    def execute_next_step(self) -> Tuple[bool, Dict[str, Any]]:
        if self.session.current_step >= len(self.session.steps):
            return False, {"status": "complete", "message": "All steps done"}
            
        step = self.session.steps[self.session.current_step]
        logger.info(f"Executing {step.query_id}: {step.purpose}")

        try:
            # Check if we have a cached result for this exact SQL
            sql_hash = f"result_{hash(step.sql)}"
            cached_result = self._get_from_cache(sql_hash, 'result')
            
            if cached_result and isinstance(cached_result, dict) and 'data' in cached_result:
                logger.info(f"Using cached result for SQL query")
                result = cached_result
            else:
                # Check if result is already cached in session context to avoid duplicate execution
                if self.session.context.get('result') is not None:
                    logger.info("Using cached SQL execution result for this query.")
                    result = self.session.context['result']
                else:
                    # Execute the SQL and cache the result
                    result = execute_nl_query(step.sql)
                    self.session.context['result'] = result
                self._add_to_cache(sql_hash, result, 'result')

            success = result.get('success', False)
            data = result.get('data', [])
            step.status = QueryStatus.COMPLETED
            step.result = {"success": success, "data": data}
            step.completed_at = datetime.utcnow()

            # 1️⃣ Hard step limit
            if len(self.session.steps) >= AnalyticsAgent.MAX_STEPS:
                logger.info("Max analysis depth reached.")
                return False, {"status": "complete", "message": "Max steps"}

            # 2️⃣ No-change detection
            sample = data[:5]
            sig = hash(json.dumps(sample, sort_keys=True, default=str))
            seen = self.session.context.setdefault('seen_sigs', set())
            if sig in seen:
                logger.info("Results converged; stopping.")
                return False, {"status": "complete", "message": "Converged"}
            seen.add(sig)

            # 3️⃣ Diminishing returns
            vals = [r.get('total_spent', 0) for r in data if 'total_spent' in r]
            if vals:
                curr = sum(vals) / len(vals)
                prev = self.session.context.get('last_avg')
                if prev and abs(curr - prev) / prev < 0.01:
                    logger.info("<1% change; halting.")
                    return False, {"status": "complete", "message": "Diminishing returns"}
                self.session.context['last_avg'] = curr

            # Generate follow-up if needed
            if success and is_insufficient(step.result):
                logger.info(f"Insufficient for {step.query_id}, generating follow-up.")
                sp = data[:5]
                pf = (
                    f"User goal: {self.session.user_question}\n"
                    f"Sample: {json.dumps(sp, default=str)}\n"
                    "What SQL next? Return only the SQL."
                )
                
                # Get follow-up SQL using prompt cache
                nxt = self._get_sql_from_prompt(pf, "follow-up analysis")
                if nxt:  # Only add if we got valid SQL
                    self.session.add_step(sql=nxt, purpose="Follow-up")

            self.session.current_step += 1
            has_next = self.session.current_step < len(self.session.steps)
            return has_next, {"status": "success", "executed": step.query_id}

        except Exception as e:
            err = str(e)
            logger.error(f"Error in execute_next_step: {err}\n{traceback.format_exc()}")
            step.status = QueryStatus.FAILED
            step.error = err
            return False, {"status": "error", "error": err, "step": step.query_id}

    def get_final_response(self) -> Dict[str, Any]:
        completed = [s for s in self.session.steps if s.status == QueryStatus.COMPLETED]
        
        # Process results to extract metrics and their data
        metrics_data = {}
        for step in completed:
            if not step.result or 'data' not in step.result:
                continue
                
            # The data comes as a list of dicts with 'metric' and 'data' keys
            for item in step.result['data']:
                if not isinstance(item, dict) or 'metric' not in item:
                    continue
                metric_name = item['metric']
                metrics_data[metric_name] = item.get('data', [])
        
        # Create a summary of the metrics
        summary = []
        for metric_name, data in metrics_data.items():
            summary.append({
                'metric': metric_name,
                'sample': data[:3] if data else []
            })
        
        # Format the data for the prompt
        prompt_data = json.dumps(summary, indent=2, default=str)
        prompt = (
            f"Based on these analytics metrics for '{self.session.user_question}', "
            f"summarize the key findings and provide 3 business recommendations. "
            f"Focus on actionable insights. Here's the data:\n{prompt_data}"
        )
        
        try:
            ans = summarize_metrics(metrics_data, self.session.user_question)
        except Exception as e:
            logger.error(f"Error generating final response: {str(e)}")
            ans = "I've analyzed your data. Here are the key metrics:"
            for item in summary:
                ans += f"\n\n{item['metric']}: {len(item['sample'])} data points"
        
        return {
            'session_id': self.session.session_id,
            'question': self.session.user_question,
            'status': 'completed',
            'metrics': metrics_data,
            'summary': summary,
            'answer': ans
        }

# Node runner

def analytics_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    if state.get('processed'):
        return {'agent_output': {'message': 'Already done', 'success': False}}
    query = state.get('query', '')
    sid = state.get('session_id', f"sess_{int(datetime.utcnow().timestamp())}")
    agent = AnalyticsAgent(session_id=sid)
    agent.analyze_question(query)
    while True:
        has_next, _ = agent.execute_next_step()
        if not has_next:
            break
    out = agent.get_final_response()
    return {'agent_output': {'success': True, 'response': out, 'data': out.get('summary'), 'session_id': sid}}
