import os
import time
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.agents import create_openai_tools_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client


@tool
def get_billing_info(query: str) -> str:
    """Retrieve billing information for the user."""
    time.sleep(0.05)
    return (
        f"Billing analysis for '{query}': "
        f"Your standard plan is $199/month. "
        f"The $299 charge includes a $100 one-time setup fee applied this month. "
        f"Future invoices will be $199/month."
    )


@tool
def get_technical_support(query: str) -> str:
    """Get technical support information."""
    time.sleep(0.05)
    return f"Technical support for '{query}': Please restart your device and check connections."


@tool
def get_general_info(query: str) -> str:
    """Get general information."""
    time.sleep(0.05)
    return f"General info for '{query}': Our hours are 9 AM - 5 PM EST, Monday-Friday."


class LangSmithAgent:
    def __init__(self):
        # 1. Ensure keys are loaded
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.tools = [get_billing_info, get_technical_support, get_general_info]
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful customer service agent."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        # 2. Specify the model explicitly
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=self.api_key)
        
        # 3. Create the agent
        agent = create_openai_tools_agent(self.llm, self.tools, self.prompt)
        
        if agent is None:
            raise ValueError("Failed to create LangChain agent. Check prompt and tool definitions.")

        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False,
            return_intermediate_steps=True
        )
        
        self.client = Client()
    
    def run(self, question: str) -> Dict[str, Any]:
        """Run the agent with automatic LangSmith tracing."""
        start_time = time.time()
        
        result = self.agent_executor.invoke({"input": question})
        
        latency = (time.time() - start_time) * 1000
        
        tools_used = []
        for step in result.get("intermediate_steps", []):
            if hasattr(step[0], 'tool'):
                tools_used.append(step[0].tool)
        
        return {
            "output": result["output"],
            "latency_ms": latency,
            "tools_used": tools_used,
            "intermediate_steps": len(result.get("intermediate_steps", [])),
            "metadata": {
                "tracing": "automatic",
                "setup_complexity": "low"
            }
        }
    
    def get_recent_traces(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Fetch recent traces for analysis."""
        try:
            runs = list(self.client.list_runs(
                project_name=os.getenv("LANGCHAIN_PROJECT", "agent-benchmark"),
                limit=limit
            ))
            
            traces = []
            for run in runs:
                traces.append({
                    "run_id": str(run.id),
                    "input": run.inputs,
                    "output": run.outputs,
                    "latency_ms": run.latency,
                    "total_tokens": run.total_tokens,
                    "status": run.status
                })
            return traces
        except Exception as e:
            return [{"error": str(e)}]