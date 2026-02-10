import os
import time
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.agents.agent import AgentExecutor
from langchain.agents import create_openai_tools_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langfuse.callback import CallbackHandler
from langfuse import Langfuse


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


class LangfuseAgent:
    def __init__(self):
        self.tools = [get_billing_info, get_technical_support, get_general_info]
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful customer service agent. 
            Classify the user's intent as Billing, Technical, or General.
            Use the appropriate tool to get information, then provide a helpful response."""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        agent = create_openai_tools_agent(self.llm, self.tools, self.prompt)
        
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False,
            return_intermediate_steps=True
        )
        
        self.langfuse = Langfuse()
    
    def run(self, question: str, session_id: str = "benchmark-session") -> Dict[str, Any]:
        """Run the agent with Langfuse tracing."""
        start_time = time.time()
        
        handler = CallbackHandler(
            session_id=session_id,
            user_id="benchmark-user",
            metadata={"environment": "benchmark", "version": "1.0"}
        )
        
        result = self.agent_executor.invoke(
            {"input": question},
            config={"callbacks": [handler]}
        )
        
        latency = (time.time() - start_time) * 1000
        
        tools_used = []
        for step in result.get("intermediate_steps", []):
            if hasattr(step[0], 'tool'):
                tools_used.append(step[0].tool)
        
        handler.flush()
        
        return {
            "output": result["output"],
            "latency_ms": latency,
            "tools_used": tools_used,
            "intermediate_steps": len(result.get("intermediate_steps", [])),
            "trace_id": handler.get_trace_id(),
            "metadata": {
                "tracing": "callback-based",
                "setup_complexity": "medium"
            }
        }
    
    def get_recent_traces(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Fetch recent traces for analysis."""
        try:
            traces_response = self.langfuse.fetch_traces(limit=limit)
            
            traces = []
            for trace in traces_response.data:
                traces.append({
                    "trace_id": trace.id,
                    "session_id": trace.session_id,
                    "input": trace.input,
                    "output": trace.output,
                    "metadata": trace.metadata
                })
            return traces
        except Exception as e:
            return [{"error": str(e)}]