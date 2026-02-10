"""
Updated Benchmark Script with RAG-based Analysis
Integrates PDF vectorization to avoid caching issues
"""
import streamlit as st
import os
import time
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
import json
from datetime import datetime
from pathlib import Path

# Import agents
from agents import LangSmithAgent, LangfuseAgent, BraintrustAgent

# Import RAG analyzer
from pdf_vectorizer import PDFVectorAnalyzer, RESEARCH_QUESTIONS

load_dotenv()

st.set_page_config(
    page_title="LLM Observability Benchmark with RAG Analysis",
    layout="wide"
)

# Configuration
PDF_PATH = "competition-health-insurance-us-markets.pdf"
INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "index_metadata.json"

st.title("LLM Agent Observability Tools Benchmark")
st.markdown("**RAG-Powered Analysis: Fresh context from health insurance market research PDF**")

st.divider()

# Sidebar: RAG Configuration
with st.sidebar:
    st.header("RAG Configuration")
    
    # Check if PDF exists
    pdf_exists = Path(PDF_PATH).exists()
    index_exists = Path(INDEX_PATH).exists()
    
    if pdf_exists:
        st.success(f"PDF found: {PDF_PATH}")
    else:
        st.error(f"PDF not found: {PDF_PATH}")
        st.info("Place the PDF in the same directory as this script")
    
    if index_exists:
        st.success("Vector index found")
        rebuild_index = st.button("Rebuild Vector Index")
    else:
        st.warning("Vector index not found")
        rebuild_index = st.button("Build Vector Index")
    
    st.divider()
    
    top_k = st.slider(
        "Context chunks per question",
        min_value=3,
        max_value=10,
        value=5,
        help="Number of PDF chunks to retrieve for each research question"
    )
    
    chunk_size = st.number_input(
        "Chunk size (characters)",
        min_value=500,
        max_value=2000,
        value=1000,
        step=100
    )
    
    chunk_overlap = st.number_input(
        "Chunk overlap (characters)",
        min_value=50,
        max_value=500,
        value=200,
        step=50
    )

# Build/Rebuild Index if requested
if rebuild_index and pdf_exists:
    with st.spinner("Building vector index from PDF..."):
        analyzer = PDFVectorAnalyzer(
            PDF_PATH,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        analyzer.load_and_split_pdf()
        analyzer.build_vector_index()
        analyzer.save_index(INDEX_PATH, METADATA_PATH)
        st.success(" Vector index built successfully!")
        st.rerun()

# Main content
st.subheader("Benchmark Configuration")

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

# Use research questions automatically
questions = RESEARCH_QUESTIONS

st.info(f"Testing with {len(RESEARCH_QUESTIONS)} research questions across {len(tools_to_test)} tools with {num_runs} run(s) each")
st.info(f"Total operations: {len(tools_to_test)} tools x {len(questions)} questions x {num_runs} runs = {len(tools_to_test) * len(questions) * num_runs}")

st.divider()

# Research Questions Section
with st.expander("View 18 Research Questions (for RAG analysis)", expanded=False):
    st.markdown("These questions will be used to benchmark the agents and analyzed using context from the health insurance market research PDF:")
    for i, q in enumerate(RESEARCH_QUESTIONS, 1):
        st.markdown(f"**{i}.** {q}")

st.divider()

# Run Benchmark Button
if st.button("Run Benchmark", type="primary", use_container_width=True):
    
    if not questions:
        st.error("No questions available!")
    elif not index_exists:
        st.error("Vector index not found! Please build the index first using the sidebar.")
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
                
                st.success(f"{tool_name} completed {len(questions)} questions x {num_runs} runs")
                
            except Exception as e:
                st.error(f"{tool_name} failed: {str(e)}")
                results[tool_name].append({"error": str(e)})
        
        progress_bar.empty()
        status_text.empty()
        
        # Save results
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
        
        st.success("Results saved to results.json")
        
        # RAG-BASED ANALYSIS
        st.divider()
        st.header("RAG-Powered Analysis")
        st.markdown("Analyzing benchmark results with fresh context from health insurance market research PDF")
        
        with st.spinner("Loading vector index and generating analyses..."):
            # Load RAG analyzer
            rag_analyzer = PDFVectorAnalyzer(PDF_PATH)
            rag_analyzer.load_index(INDEX_PATH, METADATA_PATH)
            
            # Analyze all research questions
            analyses = rag_analyzer.analyze_all_questions(
                RESEARCH_QUESTIONS,
                results,
                top_k=top_k
            )
        
        # Create tabs for results
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Performance Metrics",
            "RAG Analysis (All 18 Questions)",
            "Sample RAG Analyses",
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
                
                # Data table
                st.dataframe(perf_df, use_container_width=True)
        
        with tab2:
            st.subheader("Complete RAG Analysis - All 18 Research Questions")
            st.markdown("Each analysis uses fresh context retrieved from the PDF vector store")
            
            # Save full analysis
            full_analysis_path = f"rag_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(full_analysis_path, 'w') as f:
                json.dump(analyses, f, indent=2)
            
            st.success(f"Full analysis saved to {full_analysis_path}")
            
            # Display all analyses
            for i, (question, analysis) in enumerate(analyses.items(), 1):
                with st.expander(f"Q{i}: {question}", expanded=False):
                    st.markdown(analysis)
        
        with tab3:
            st.subheader("Sample RAG Analyses (First 3 Questions)")
            
            for i, (question, analysis) in enumerate(list(analyses.items())[:3], 1):
                st.markdown(f"### Question {i}")
                st.info(question)
                st.markdown("**Analysis:**")
                st.markdown(analysis)
                st.divider()
        
        with tab4:
            st.subheader("Detailed Run Results")
            
            for tool_name in tools_to_test:
                with st.expander(f"{tool_name} - All Runs", expanded=False):
                    runs = results[tool_name]
                    
                    if runs and "error" not in runs[0]:
                        for idx, run in enumerate(runs):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Agent Latency", f"{run['latency_ms']:.2f} ms")
                            
                            with col2:
                                st.metric("Total Time", f"{run['total_time']:.2f} ms")
                            
                            with col3:
                                st.metric("Tools Used", len(run.get('tools_used', [])))
                            
                            st.markdown("**Output:**")
                            st.info(run['output'])
                            
                            if idx < len(runs) - 1:
                                st.divider()
                    else:
                        st.error(f"Error: {runs[0].get('error', 'Unknown error')}")
        
        with tab5:
            st.subheader("Export Benchmark Data")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    label="Download Benchmark Results (JSON)",
                    data=json.dumps(results_data, indent=2),
                    file_name=f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col2:
                if perf_data:
                    csv_data = pd.DataFrame(perf_data).to_csv(index=False)
                    st.download_button(
                        label="Download Performance CSV",
                        data=csv_data,
                        file_name=f"performance_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col3:
                with open(full_analysis_path, 'r') as f:
                    analysis_json = f.read()
                
                st.download_button(
                    label="Download RAG Analysis (JSON)",
                    data=analysis_json,
                    file_name=full_analysis_path,
                    mime="application/json",
                    use_container_width=True
                )
            
            st.subheader("Results Preview")
            st.json(results_data, expanded=False)

else:
    st.info("Configure your test parameters above and click 'Run Benchmark' to start")
    
    st.markdown("""
    ### How This Works:
    
    1. **Agent Benchmarking**: Test LLM observability tools (LangSmith, Langfuse, Braintrust) with 18 research questions
    2. **PDF Vectorization**: Health insurance market research PDF is chunked and embedded using FAISS
    3. **RAG Analysis**: For each research question, retrieve relevant PDF context and analyze benchmark results
    4. **Fresh Context**: Each analysis retrieves new context from the vector store to avoid caching
    5. **18 Research Questions**: Comprehensive analysis covering market power, methodology, findings, and policy
    
    **No hardcoded conclusions** - All insights are generated from actual benchmark data + PDF context.
    """)

st.divider()
st.caption("Powered by RAG | Fresh context on every run | No caching issues")