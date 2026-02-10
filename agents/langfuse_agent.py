import os
import time
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langfuse.langchain import CallbackHandler
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
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.tools = [get_billing_info, get_technical_support, get_general_info]
        
        # Use bind_tools for direct tool calling
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=self.api_key)
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Initialize Langfuse client
        self.langfuse = Langfuse()
    
    def run(self, question: str, session_id: str = "benchmark-session") -> Dict[str, Any]:
        """Run the agent with Langfuse tracing."""
        start_time = time.time()
        
        # Minimal CallbackHandler - no parameters, no methods
        handler = CallbackHandler()
        
        # Simple agentic loop
        messages = [{"role": "user", "content": question}]
        tools_used = []
        intermediate_steps = 0
        
        # Allow up to 5 iterations
        for _ in range(5):
            response = self.llm_with_tools.invoke(
                messages,
                config={"callbacks": [handler]}
            )
            
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
        
        # Langfuse SDK v3: CallbackHandler has no methods
        # - No flush() method
        # - No get_trace_id() method
        # Traces are sent automatically
        
        return {
            "output": final_output,
            "latency_ms": latency,
            "tools_used": list(set(tools_used)),
            "intermediate_steps": intermediate_steps,
            "trace_id": "auto-traced",  # v3 doesn't expose trace ID from handler
            "metadata": {
                "tracing": "callback-based",
                "setup_complexity": "medium",
                "sdk_version": "v3",
                "note": "Traces sent automatically, check Langfuse UI"
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