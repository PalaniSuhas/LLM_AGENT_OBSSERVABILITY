import os
import time
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langsmith import Client
import json


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
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.tools = [get_billing_info, get_technical_support, get_general_info]
        
        # Use bind_tools for direct tool calling
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=self.api_key)
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        self.client = Client()
    
    def run(self, question: str) -> Dict[str, Any]:
        """Run the agent with automatic LangSmith tracing."""
        start_time = time.time()
        
        # Simple agentic loop
        messages = [{"role": "user", "content": question}]
        tools_used = []
        intermediate_steps = 0
        
        # Allow up to 5 iterations
        for _ in range(5):
            response = self.llm_with_tools.invoke(messages)
            
            # Check if there are tool calls
            if hasattr(response, 'tool_calls') and response.tool_calls:
                intermediate_steps += 1
                
                # Execute each tool call
                for tool_call in response.tool_calls:
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']
                    
                    # Find and execute the tool
                    tool_func = next((t for t in self.tools if t.name == tool_name), None)
                    if tool_func:
                        tools_used.append(tool_name)
                        result = tool_func.invoke(tool_args)
                        
                        # Add tool result to messages
                        messages.append({
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [tool_call]
                        })
                        messages.append({
                            "role": "tool",
                            "content": result,
                            "tool_call_id": tool_call.get('id', 'default')
                        })
                
                # Continue the loop to get final response
                continue
            else:
                # No more tool calls, we have the final answer
                final_output = response.content
                break
        else:
            final_output = "Max iterations reached"
        
        latency = (time.time() - start_time) * 1000
        
        return {
            "output": final_output,
            "latency_ms": latency,
            "tools_used": list(set(tools_used)),
            "intermediate_steps": intermediate_steps,
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