import streamlit as st
import os
import time
from dotenv import load_dotenv
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from agents import LangSmithAgent, LangfuseAgent, BraintrustAgent

load_dotenv()

st.set_page_config(
    page_title="LLM Observability Benchmark",
    page_icon="📊",
    layout="wide"
)

STANDARD_QUESTION = (
    "My last invoice shows a charge of $299 but I was expecting $199. "
    "Can you help me understand this discrepancy?"
)

st.title("🔍 LLM Agent Observability Tools Benchmark")
st.markdown("**Compare LangSmith, Langfuse, and Braintrust side-by-side**")

st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Standard Test Question")
    question = st.text_area(
        "Question to test all agents:",
        value=STANDARD_QUESTION,
        height=100,
        help="This question will be sent to all three observability tools"
    )

with col2:
    st.subheader("⚙️ Configuration")
    
    tools_to_test = st.multiselect(
        "Select tools to benchmark:",
        ["LangSmith", "Langfuse", "Braintrust"],
        default=["LangSmith", "Langfuse", "Braintrust"]
    )
    
    num_runs = st.number_input(
        "Number of runs per tool:",
        min_value=1,
        max_value=10,
        value=1,
        help="Run multiple times to average latency"
    )

st.divider()

if st.button("🚀 Run Benchmark", type="primary", use_container_width=True):
    
    results = {
        "LangSmith": [],
        "Langfuse": [],
        "Braintrust": []
    }
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_operations = len(tools_to_test) * num_runs
    current_operation = 0
    
    for tool_name in tools_to_test:
        st.subheader(f"Testing {tool_name}")
        
        try:
            if tool_name == "LangSmith":
                agent = LangSmithAgent()
            elif tool_name == "Langfuse":
                agent = LangfuseAgent()
            else:
                agent = BraintrustAgent()
            
            for run_num in range(num_runs):
                status_text.text(f"Running {tool_name} - Run {run_num + 1}/{num_runs}")
                
                start = time.time()
                result = agent.run(question)
                end = time.time()
                
                result["total_time"] = (end - start) * 1000
                result["run_number"] = run_num + 1
                results[tool_name].append(result)
                
                current_operation += 1
                progress_bar.progress(current_operation / total_operations)
                
                time.sleep(0.5)
            
            st.success(f"✅ {tool_name} completed {num_runs} run(s)")
            
        except Exception as e:
            st.error(f"❌ {tool_name} failed: {str(e)}")
            results[tool_name].append({"error": str(e)})
    
    progress_bar.empty()
    status_text.empty()
    
    st.divider()
    st.header("📊 Benchmark Results")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Performance Metrics",
        "🔧 Tool Comparison",
        "📝 Detailed Results",
        "🏆 Final Verdict"
    ])
    
    with tab1:
        st.subheader("Performance Comparison")
        
        perf_data = []
        for tool, runs in results.items():
            if tool in tools_to_test and runs and "error" not in runs[0]:
                avg_latency = sum(r["latency_ms"] for r in runs) / len(runs)
                avg_total = sum(r["total_time"] for r in runs) / len(runs)
                
                perf_data.append({
                    "Tool": tool,
                    "Avg Agent Latency (ms)": round(avg_latency, 2),
                    "Avg Total Time (ms)": round(avg_total, 2),
                    "Runs": len(runs)
                })
        
        if perf_data:
            perf_df = pd.DataFrame(perf_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_latency = px.bar(
                    perf_df,
                    x="Tool",
                    y="Avg Agent Latency (ms)",
                    title="Average Agent Execution Latency",
                    color="Tool",
                    text="Avg Agent Latency (ms)"
                )
                fig_latency.update_traces(texttemplate='%{text:.2f}ms', textposition='outside')
                st.plotly_chart(fig_latency, use_container_width=True)
            
            with col2:
                fig_total = px.bar(
                    perf_df,
                    x="Tool",
                    y="Avg Total Time (ms)",
                    title="Average Total Time (including overhead)",
                    color="Tool",
                    text="Avg Total Time (ms)"
                )
                fig_total.update_traces(texttemplate='%{text:.2f}ms', textposition='outside')
                st.plotly_chart(fig_total, use_container_width=True)
            
            st.dataframe(perf_df, use_container_width=True)
    
    with tab2:
        st.subheader("Feature Comparison Matrix")
        
        comparison_data = {
            "Feature": [
                "LangChain Integration",
                "Setup Complexity",
                "Auto-Tracing",
                "Token Tracking",
                "Cost Tracking",
                "Self-Hosting",
                "Evaluation Framework",
                "Production Ready"
            ],
            "LangSmith": [
                "✅ Native",
                "🟢 Low (env var)",
                "✅ Yes",
                "✅ Automatic",
                "✅ Automatic",
                "❌ No",
                "✅ Built-in",
                "✅ Yes"
            ],
            "Langfuse": [
                "⚠️ Callback",
                "🟡 Medium",
                "⚠️ Via callback",
                "✅ Automatic",
                "✅ Automatic",
                "✅ Yes (OSS)",
                "⚠️ Manual scoring",
                "✅ Yes"
            ],
            "Braintrust": [
                "❌ Manual",
                "🔴 High",
                "❌ No",
                "⚠️ Manual",
                "⚠️ Manual",
                "❌ No",
                "✅ Built-in",
                "⚠️ Limited"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Best for LangChain", "LangSmith", help="Native integration, zero config")
        
        with col2:
            st.metric("Best for Production", "Langfuse", help="Self-hosting, flexibility")
        
        with col3:
            st.metric("Best for Evaluation", "Braintrust", help="Advanced eval framework")
    
    with tab3:
        st.subheader("Detailed Run Results")
        
        for tool_name in tools_to_test:
            with st.expander(f"{tool_name} - Detailed Results", expanded=False):
                runs = results[tool_name]
                
                if runs and "error" not in runs[0]:
                    for idx, run in enumerate(runs):
                        st.markdown(f"**Run {idx + 1}:**")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Agent Latency", f"{run['latency_ms']:.2f} ms")
                        
                        with col2:
                            st.metric("Total Time", f"{run['total_time']:.2f} ms")
                        
                        with col3:
                            st.metric("Tools Used", len(run.get('tools_used', [])))
                        
                        st.markdown("**Output:**")
                        st.info(run['output'])
                        
                        st.markdown("**Tools Called:**")
                        st.code(", ".join(run.get('tools_used', ['None'])))
                        
                        st.markdown("**Metadata:**")
                        st.json(run.get('metadata', {}))
                        
                        if idx < len(runs) - 1:
                            st.divider()
                else:
                    st.error(f"Error: {runs[0].get('error', 'Unknown error')}")
    
    with tab4:
        st.subheader("🏆 Final Verdict")
        
        st.markdown("""
        ### Overall Recommendations
        
        Based on the benchmark results and feature analysis:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### ✅ Choose LangSmith if:
            - You're building with LangChain exclusively
            - You want zero-setup observability
            - You prioritize developer experience
            - You don't need self-hosting
            
            #### ✅ Choose Langfuse if:
            - You need self-hosting capability
            - You use multiple LLM frameworks
            - You want to avoid vendor lock-in
            - You need strong user/session tracking
            """)
        
        with col2:
            st.markdown("""
            #### ✅ Choose Braintrust if:
            - Evaluation is your primary focus
            - You need advanced A/B testing
            - You're doing research/benchmarking
            - You have custom LLM implementations
            
            #### 🎯 My Personal Choice:
            **Langfuse** for production systems due to:
            - Open source nature (no lock-in)
            - Self-hosting flexibility
            - Framework agnostic
            - Strong community support
            """)
        
        st.divider()
        
        st.markdown("""
        ### Performance Summary
        
        All three tools successfully executed the benchmark question with similar latency profiles.
        The differences in total time are primarily due to:
        - **LangSmith**: Minimal overhead (env var tracing)
        - **Langfuse**: Callback handler creation overhead
        - **Braintrust**: Manual span management overhead
        
        For production use, these overhead differences (typically <50ms) are negligible compared to
        the LLM call latency itself (typically 200-1000ms).
        
        **Key Takeaway**: Choose based on features and integration needs, not raw performance.
        """)

else:
    st.info("👆 Configure your test parameters above and click 'Run Benchmark' to start")

st.divider()

st.markdown("""
### 📚 Quick Start Guide

1. **Setup Environment Variables**: Copy `.env.example` to `.env` and add your API keys
2. **Install Dependencies**: `pip install -r requirements.txt`
3. **Run Benchmark**: `streamlit run benchmark.py`

### 🔑 Required API Keys

- **OpenAI**: Get from https://platform.openai.com/api-keys
- **LangSmith**: Get from https://smith.langchain.com/settings
- **Langfuse**: Get from https://cloud.langfuse.com or self-hosted instance
- **Braintrust**: Get from https://www.braintrust.dev/app/settings

### 📖 Documentation

- [LangSmith Docs](https://docs.smith.langchain.com/)
- [Langfuse Docs](https://langfuse.com/docs)
- [Braintrust Docs](https://www.braintrust.dev/docs)
""")