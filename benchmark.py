import streamlit as st
import os
import time
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
import json
from datetime import datetime
from agents import LangSmithAgent, LangfuseAgent, BraintrustAgent
from llm_analyzer import BenchmarkAnalyzer

load_dotenv()

st.set_page_config(
    page_title="LLM Observability Benchmark",
    page_icon="",
    layout="wide"
)

# Default test questions
DEFAULT_QUESTIONS = [
    "My last invoice shows a charge of $299 but I was expecting $199. Can you help me understand this discrepancy?",
    "I need technical support for setting up my new device. Can you guide me through the process?",
    "What are your business hours and how can I contact customer support?",
    "I want to upgrade my plan. What options do I have and what are the costs?"
]

st.title("LLM Agent Observability Tools Benchmark")
st.markdown("**AI-Powered Analysis: LLM decides the winner based on real benchmark data**")

st.divider()

# Configuration Section
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Test Questions")
    
    # Allow users to add/edit questions
    num_questions = st.number_input(
        "Number of test questions:",
        min_value=1,
        max_value=10,
        value=4,
        help="Test agents with multiple questions"
    )
    
    questions = []
    for i in range(num_questions):
        default_q = DEFAULT_QUESTIONS[i] if i < len(DEFAULT_QUESTIONS) else ""
        q = st.text_area(
            f"Question {i+1}:",
            value=default_q,
            height=80,
            key=f"question_{i}"
        )
        if q.strip():
            questions.append(q.strip())

with col2:
    st.subheader("Configuration")
    
    tools_to_test = st.multiselect(
        "Select tools to benchmark:",
        ["LangSmith", "Langfuse", "Braintrust"],
        default=["LangSmith", "Langfuse", "Braintrust"]
    )
    
    num_runs = st.number_input(
        "Runs per question per tool:",
        min_value=1,
        max_value=5,
        value=1,
        help="Run each question multiple times to average latency"
    )
    
    st.info(f"Total operations: {len(tools_to_test)} tools × {len(questions)} questions × {num_runs} runs = {len(tools_to_test) * len(questions) * num_runs}")

st.divider()

# Run Benchmark Button
if st.button(" Run Benchmark", type="primary", width='stretch'):
    
    if not questions:
        st.error("Please add at least one test question!")
    else:
        # Initialize results storage
        results = {
            "LangSmith": [],
            "Langfuse": [],
            "Braintrust": []
        }
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_operations = len(tools_to_test) * len(questions) * num_runs
        current_operation = 0
        
        # Run benchmarks
        for tool_name in tools_to_test:
            st.subheader(f"Testing {tool_name}")
            
            try:
                # Initialize agent
                if tool_name == "LangSmith":
                    agent = LangSmithAgent()
                elif tool_name == "Langfuse":
                    agent = LangfuseAgent()
                else:
                    agent = BraintrustAgent()
                
                # Test each question
                for q_idx, question in enumerate(questions):
                    for run_num in range(num_runs):
                        status_text.text(f"Running {tool_name} - Question {q_idx + 1}/{len(questions)} - Run {run_num + 1}/{num_runs}")
                        
                        start = time.time()
                        result = agent.run(question)
                        end = time.time()
                        
                        result["total_time"] = (end - start) * 1000
                        result["run_number"] = run_num + 1
                        result["question_index"] = q_idx
                        result["question"] = question
                        results[tool_name].append(result)
                        
                        current_operation += 1
                        progress_bar.progress(current_operation / total_operations)
                        
                        time.sleep(0.3)
                
                st.success(f"✓ {tool_name} completed {len(questions)} questions × {num_runs} runs")
                
            except Exception as e:
                st.error(f"✗ {tool_name} failed: {str(e)}")
                results[tool_name].append({"error": str(e)})
        
        progress_bar.empty()
        status_text.empty()
        
        # Save results to JSON
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "tools_tested": tools_to_test,
                "questions": questions,
                "num_runs": num_runs
            },
            "results": results
        }
        
        with open("results.json", "w") as f:
            json.dump(results_data, f, indent=2)
        
        st.success("✓ Results saved to results.json")
        
        # Initialize LLM analyzer
        st.divider()
        st.header("🤖 AI-Powered Analysis")
        
        with st.spinner("LLM is analyzing benchmark results..."):
            analyzer = BenchmarkAnalyzer()
            analysis = analyzer.analyze_results(results, tools_to_test, questions)
        
        # Create tabs for results
        tab1, tab2, tab3, tab4 = st.tabs([
            "Performance Metrics",
            "LLM Analysis",
            "Detailed Results",
            "Export Data"
        ])
        
        with tab1:
            st.subheader("Performance Comparison")
            
            perf_data = []
            for tool, runs in results.items():
                if tool in tools_to_test and runs and "error" not in runs[0]:
                    valid_runs = [r for r in runs if "error" not in r]
                    
                    if valid_runs:
                        avg_latency = sum(r["latency_ms"] for r in valid_runs) / len(valid_runs)
                        avg_total = sum(r["total_time"] for r in valid_runs) / len(valid_runs)
                        min_latency = min(r["latency_ms"] for r in valid_runs)
                        max_latency = max(r["latency_ms"] for r in valid_runs)
                        
                        perf_data.append({
                            "Tool": tool,
                            "Avg Latency (ms)": round(avg_latency, 2),
                            "Avg Total Time (ms)": round(avg_total, 2),
                            "Min Latency (ms)": round(min_latency, 2),
                            "Max Latency (ms)": round(max_latency, 2),
                            "Total Runs": len(valid_runs)
                        })
            
            if perf_data:
                perf_df = pd.DataFrame(perf_data)
                
                # Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_latency = px.bar(
                        perf_df,
                        x="Tool",
                        y="Avg Latency (ms)",
                        title="Average Agent Execution Latency",
                        color="Tool",
                        text="Avg Latency (ms)"
                    )
                    fig_latency.update_traces(texttemplate='%{text:.2f}ms', textposition='outside')
                    st.plotly_chart(fig_latency, width='stretch')
                
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
                    st.plotly_chart(fig_total, width='stretch')
                
                # Data table
                st.dataframe(perf_df, width='stretch')
                
                # LLM-generated summary metrics
                st.subheader("Key Insights")
                try:
                    metrics = analyzer.generate_summary_metrics(results, tools_to_test)
                    cols = st.columns(len(metrics))
                    for idx, metric in enumerate(metrics):
                        with cols[idx]:
                            st.metric(
                                metric.get("label", "Metric"),
                                metric.get("value", "N/A"),
                                help=metric.get("description", "")
                            )
                except Exception as e:
                    st.warning(f"Could not generate metrics: {e}")
        
        with tab2:
            st.subheader("Complete LLM Analysis")
            st.markdown("**All conclusions are generated by AI based on actual benchmark results - nothing is hardcoded.**")
            
            st.divider()
            
            # Display LLM analysis
            st.markdown(analysis["raw_analysis"])
            
            st.divider()
            
            # LLM-generated comparison matrix
            st.subheader("Feature Comparison")
            try:
                comparison_table = analyzer.generate_comparison_matrix(results, tools_to_test)
                st.markdown(comparison_table)
            except Exception as e:
                st.warning(f"Could not generate comparison matrix: {e}")
        
        with tab3:
            st.subheader("Detailed Run Results")
            
            for tool_name in tools_to_test:
                with st.expander(f"{tool_name} - All Runs", expanded=False):
                    runs = results[tool_name]
                    
                    if runs and "error" not in runs[0]:
                        # Group by question
                        questions_dict = {}
                        for run in runs:
                            q_idx = run.get("question_index", 0)
                            if q_idx not in questions_dict:
                                questions_dict[q_idx] = []
                            questions_dict[q_idx].append(run)
                        
                        for q_idx, q_runs in questions_dict.items():
                            st.markdown(f"**Question {q_idx + 1}:** {q_runs[0]['question'][:100]}...")
                            
                            for idx, run in enumerate(q_runs):
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Agent Latency", f"{run['latency_ms']:.2f} ms")
                                
                                with col2:
                                    st.metric("Total Time", f"{run['total_time']:.2f} ms")
                                
                                with col3:
                                    st.metric("Tools Used", len(run.get('tools_used', [])))
                                
                                st.markdown("**Output:**")
                                st.info(run['output'])
                                
                                if idx < len(q_runs) - 1:
                                    st.divider()
                            
                            st.markdown("---")
                    else:
                        st.error(f"Error: {runs[0].get('error', 'Unknown error')}")
        
        with tab4:
            st.subheader("Export Benchmark Data")
            
            # JSON download
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label=" Download Full Results (JSON)",
                    data=json.dumps(results_data, indent=2),
                    file_name=f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    width='stretch'
                )
            
            with col2:
                # CSV export of performance data
                if perf_data:
                    csv_data = pd.DataFrame(perf_data).to_csv(index=False)
                    st.download_button(
                        label="Download Performance Summary (CSV)",
                        data=csv_data,
                        file_name=f"performance_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        width='stretch'
                    )
            
            # Show preview
            st.subheader("Results Preview")
            st.json(results_data, expanded=False)

else:
    st.info(" Configure your test parameters above and click 'Run Benchmark' to start")
    
    st.markdown("""
    ### How it works:
    
    1. **Multiple Test Questions**: Test agents with diverse queries to get comprehensive results
    2. **Multiple Runs**: Average performance across multiple runs for accuracy
    3. **AI-Powered Analysis**: An LLM analyzes the actual benchmark data and decides the winner
    4. **No Hardcoded Recommendations**: All conclusions are generated dynamically based on real results
    5. **Complete Export**: Download all results in JSON format for further analysis
    
    The LLM will analyze performance, integration complexity, and provide a single definitive recommendation.
    """)

st.divider()
st.caption(" Powered by LLM-driven analysis | No hardcoded conclusions")