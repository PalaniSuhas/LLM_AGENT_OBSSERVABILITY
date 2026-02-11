"""
RAG Observability Benchmark - Compare LangSmith, Langfuse, and Braintrust
Answers 18 research questions using RAG and measures observability tool performance
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

from pdf_vectorizer import PDFVectorAnalyzer, RESEARCH_QUESTIONS

load_dotenv()

st.set_page_config(
    page_title="RAG Observability Benchmark",
    layout="wide"
)

PDF_PATH = "competition-health-insurance-us-markets.pdf"
INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "index_metadata.json"

st.title("RAG Observability Tools Benchmark")
st.markdown("**Compare LangSmith, Langfuse, and Braintrust on 18 health insurance research questions**")

st.divider()

with st.sidebar:
    st.header("Configuration")
    
    pdf_exists = Path(PDF_PATH).exists()
    index_exists = Path(INDEX_PATH).exists()
    
    if pdf_exists:
        st.success(f"PDF found")
    else:
        st.error(f"PDF not found")
    
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
        value=5
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
        st.success("Vector index built successfully")
        st.rerun()

st.subheader("Benchmark Configuration")

tools_to_test = st.multiselect(
    "Select observability tools to benchmark:",
    ["LangSmith", "Langfuse", "Braintrust"],
    default=["LangSmith", "Langfuse", "Braintrust"]
)

num_runs = st.number_input(
    "Runs per question per tool:",
    min_value=1,
    max_value=5,
    value=1
)

st.info(f"Testing {len(RESEARCH_QUESTIONS)} research questions across {len(tools_to_test)} tools with {num_runs} run(s) each")
st.info(f"Total operations: {len(tools_to_test)} x {len(RESEARCH_QUESTIONS)} x {num_runs} = {len(tools_to_test) * len(RESEARCH_QUESTIONS) * num_runs}")

st.divider()

with st.expander("View All 18 Research Questions", expanded=False):
    for i, q in enumerate(RESEARCH_QUESTIONS, 1):
        st.markdown(f"**{i}.** {q}")

st.divider()

if st.button("Run Benchmark", type="primary", width='stretch'):
    
    if not index_exists:
        st.error("Vector index not found. Please build the index first.")
    else:
        results = {
            "LangSmith": [],
            "Langfuse": [],
            "Braintrust": []
        }
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_operations = len(tools_to_test) * len(RESEARCH_QUESTIONS) * num_runs
        current_operation = 0
        
        st.header("Running Benchmark")
        
        for tool_name in tools_to_test:
            st.subheader(f"Testing {tool_name}")
            
            try:
                analyzer = PDFVectorAnalyzer(PDF_PATH)
                analyzer.load_index(INDEX_PATH, METADATA_PATH)
                
                for q_idx, question in enumerate(RESEARCH_QUESTIONS):
                    for run_num in range(num_runs):
                        status_text.text(f"Running {tool_name} - Question {q_idx + 1}/{len(RESEARCH_QUESTIONS)} - Run {run_num + 1}/{num_runs}")
                        
                        start = time.time()
                        
                        if tool_name == "LangSmith":
                            os.environ["LANGCHAIN_TRACING_V2"] = "true"
                            answer = analyzer.answer_single_question(question, top_k=top_k)
                            os.environ["LANGCHAIN_TRACING_V2"] = "false"
                        elif tool_name == "Langfuse":
                            from langfuse.langchain import CallbackHandler
                            handler = CallbackHandler()
                            
                            contexts = analyzer.retrieve_context(question, top_k=top_k)
                            combined_context = "\n\n---\n\n".join(contexts)
                            
                            system_prompt = """You are an expert analyst studying health insurance market competition.

You will be provided with:
1. A research question about health insurance markets
2. Relevant excerpts from a research report on health insurance competition

Your task is to answer the question thoroughly using ONLY the information provided in the context.

Guidelines:
- Base your answer entirely on the provided context
- Quote specific data, statistics, or findings when available
- If the context doesn't contain enough information to fully answer the question, acknowledge this
- Be comprehensive but concise
- Use clear, professional language
- Cite specific facts and figures from the report when relevant"""

                            user_prompt = f"""**Context from Health Insurance Market Report:**

{combined_context}

**Research Question:**
{question}

**Your Answer:**"""

                            from langchain.schema import HumanMessage, SystemMessage
                            messages = [
                                SystemMessage(content=system_prompt),
                                HumanMessage(content=user_prompt)
                            ]
                            
                            response = analyzer.llm.invoke(messages, config={"callbacks": [handler]})
                            answer = response.content
                        else:
                            import braintrust
                            logger = braintrust.init(project="rag-benchmark")
                            span = logger.start_span(name="rag_question", input={"question": question})
                            answer = analyzer.answer_single_question(question, top_k=top_k)
                            span.log(output=answer)
                            span.end()
                            logger.flush()
                        
                        end = time.time()
                        
                        result = {
                            "question": question,
                            "answer": answer,
                            "latency_ms": (end - start) * 1000,
                            "run_number": run_num + 1,
                            "question_index": q_idx + 1,
                            "top_k": top_k
                        }
                        
                        results[tool_name].append(result)
                        
                        current_operation += 1
                        progress_bar.progress(current_operation / total_operations)
                        
                        time.sleep(0.3)
                
                st.success(f"{tool_name} completed {len(RESEARCH_QUESTIONS)} questions x {num_runs} runs")
                
            except Exception as e:
                st.error(f"{tool_name} failed: {str(e)}")
                results[tool_name].append({"error": str(e)})
        
        progress_bar.empty()
        status_text.empty()
        
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "tools_tested": tools_to_test,
                "questions": RESEARCH_QUESTIONS,
                "num_runs": num_runs,
                "top_k": top_k,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            },
            "results": results
        }
        
        output_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(results_data, f, indent=2)
        
        st.success(f"Benchmark results saved to {output_file}")
        
        st.divider()
        st.header("Results")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "Performance Metrics",
            "Answer Quality Comparison",
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
                        min_latency = min(r["latency_ms"] for r in valid_runs)
                        max_latency = max(r["latency_ms"] for r in valid_runs)
                        
                        perf_data.append({
                            "Tool": tool,
                            "Avg Latency (ms)": round(avg_latency, 2),
                            "Min Latency (ms)": round(min_latency, 2),
                            "Max Latency (ms)": round(max_latency, 2),
                            "Total Runs": len(valid_runs)
                        })
            
            if perf_data:
                perf_df = pd.DataFrame(perf_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_latency = px.bar(
                        perf_df,
                        x="Tool",
                        y="Avg Latency (ms)",
                        title="Average RAG Query Latency",
                        color="Tool",
                        text="Avg Latency (ms)"
                    )
                    fig_latency.update_traces(texttemplate='%{text:.2f}ms', textposition='outside')
                    st.plotly_chart(fig_latency, width='stretch')
                
                with col2:
                    fig_range = px.bar(
                        perf_df,
                        x="Tool",
                        y=["Min Latency (ms)", "Max Latency (ms)"],
                        title="Latency Range",
                        barmode="group"
                    )
                    st.plotly_chart(fig_range, width='stretch')
                
                st.dataframe(perf_df, width='stretch')
        
        with tab2:
            st.subheader("Answer Quality Comparison")
            
            question_selector = st.selectbox(
                "Select a question to compare answers:",
                range(len(RESEARCH_QUESTIONS)),
                format_func=lambda x: f"Q{x+1}: {RESEARCH_QUESTIONS[x][:80]}..."
            )
            
            selected_question = RESEARCH_QUESTIONS[question_selector]
            st.info(selected_question)
            
            for tool_name in tools_to_test:
                tool_results = [r for r in results[tool_name] if r.get("question") == selected_question]
                if tool_results:
                    st.markdown(f"### {tool_name}")
                    st.markdown(tool_results[0]["answer"])
                    st.caption(f"Latency: {tool_results[0]['latency_ms']:.2f}ms")
                    st.divider()
        
        with tab3:
            st.subheader("Detailed Run Results")
            
            for tool_name in tools_to_test:
                with st.expander(f"{tool_name} - All Runs", expanded=False):
                    runs = results[tool_name]
                    
                    if runs and "error" not in runs[0]:
                        for idx, run in enumerate(runs):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Question", f"#{run['question_index']}")
                            
                            with col2:
                                st.metric("Latency", f"{run['latency_ms']:.2f} ms")
                            
                            with col3:
                                st.metric("Run", f"#{run['run_number']}")
                            
                            st.markdown("**Question:**")
                            st.info(run['question'])
                            
                            st.markdown("**Answer:**")
                            st.markdown(run['answer'])
                            
                            if idx < len(runs) - 1:
                                st.divider()
                    else:
                        st.error(f"Error: {runs[0].get('error', 'Unknown error')}")
        
        with tab4:
            st.subheader("Export Benchmark Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="Download Complete Results (JSON)",
                    data=json.dumps(results_data, indent=2),
                    file_name=output_file,
                    mime="application/json",
                    width='stretch'
                )
            
            with col2:
                if perf_data:
                    csv_data = pd.DataFrame(perf_data).to_csv(index=False)
                    st.download_button(
                        label="Download Performance CSV",
                        data=csv_data,
                        file_name=f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        width='stretch'
                    )
            
            st.subheader("Results Preview")
            st.json(results_data, expanded=False)

else:
    st.info("Configure your test parameters above and click 'Run Benchmark' to start")
    
    st.markdown("""
    ### How This Works:
    
    **RAG-Based Benchmarking**
    - Tests 3 observability tools: LangSmith, Langfuse, Braintrust
    - Uses 18 research questions about health insurance markets
    - Measures latency, answer quality, and observability features
    
    **Process:**
    1. Load vector index from PDF
    2. For each tool and question:
       - Retrieve relevant context chunks
       - Generate answer using LLM
       - Track with respective observability tool
    3. Compare performance metrics
    
    **Key Metrics:**
    - Average query latency
    - Answer consistency across runs
    - Observability tool overhead
    """)

st.divider()