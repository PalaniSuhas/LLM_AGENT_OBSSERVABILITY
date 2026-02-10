import os
import time
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.tools import tool


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


class BraintrustAgent:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.tools = [get_billing_info, get_technical_support, get_general_info]
        
        # Use bind_tools for direct tool calling
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=self.api_key)
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Check for Braintrust API key
        braintrust_key = os.getenv("BRAINTRUST_API_KEY")
        if not braintrust_key:
            self.logger = None
            self.skip_tracing = True
        else:
            try:
                import braintrust
                self.logger = braintrust.init(project="agent-benchmark")
                self.skip_tracing = False
            except Exception as e:
                print(f"Warning: Braintrust initialization failed: {e}")
                self.logger = None
                self.skip_tracing = True
    
    def run(self, question: str) -> Dict[str, Any]:
        """Run the agent with optional Braintrust tracing."""
        start_time = time.time()
        
        # Start span only if Braintrust is configured
        span = None
        if not self.skip_tracing and self.logger:
            try:
                span = self.logger.start_span(
                    name="agent_execution",
                    input={"question": question}
                )
            except Exception as e:
                print(f"Warning: Failed to start Braintrust span: {e}")
        
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
        
        # Log to Braintrust if configured
        if span:
            try:
                span.log(
                    output=final_output,
                    metadata={
                        "latency_ms": latency,
                        "tools_used": tools_used,
                        "intermediate_steps": intermediate_steps
                    }
                )
                span.end()
                self.logger.flush()
            except Exception as e:
                print(f"Warning: Failed to log to Braintrust: {e}")
        
        return {
            "output": final_output,
            "latency_ms": latency,
            "tools_used": list(set(tools_used)),
            "intermediate_steps": intermediate_steps,
            "metadata": {
                "tracing": "manual" if not self.skip_tracing else "disabled (no API key)",
                "setup_complexity": "high"
            }
        }
    
    def get_recent_traces(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Fetch recent traces for analysis."""
        if self.skip_tracing:
            return [{"info": "Braintrust tracing disabled - no API key configured"}]
        return [{"info": "Braintrust trace retrieval requires project API access"}]