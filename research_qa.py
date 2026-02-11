"""
Health Insurance Market Research - RAG Q&A System
Answer 18 research questions using PDF context only
"""
import streamlit as st
import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from pdf_vectorizer import PDFVectorAnalyzer, RESEARCH_QUESTIONS

load_dotenv()

st.set_page_config(
    page_title="Health Insurance Market Research Q&A",
    layout="wide",
    initial_sidebar_state="expanded"
)

PDF_PATH = "competition-health-insurance-us-markets.pdf"
INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "index_metadata.json"

st.title("Health Insurance Market Research Q&A")
st.markdown("**Answer 18 research questions using RAG from the health insurance competition PDF**")

st.divider()

with st.sidebar:
    st.header("RAG Configuration")
    
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
    
    st.subheader("Retrieval Settings")
    
    top_k = st.slider(
        "Context chunks per question",
        min_value=3,
        max_value=15,
        value=8
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
    
    st.divider()
    
    st.subheader("About")
    st.markdown(f"""
    **Questions:** {len(RESEARCH_QUESTIONS)}  
    **Source:** Health Insurance Market PDF  
    **Method:** FAISS + OpenAI Embeddings  
    """)

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

st.subheader("Research Questions")

with st.expander("View All 18 Research Questions", expanded=False):
    for i, q in enumerate(RESEARCH_QUESTIONS, 1):
        st.markdown(f"**{i}.** {q}")

st.divider()

if st.button("Answer All Questions with RAG", type="primary", width='stretch'):
    
    if not index_exists:
        st.error("Vector index not found. Please build the index first.")
    elif not pdf_exists:
        st.error("PDF not found. Please add the PDF file to the directory.")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("Loading vector index..."):
            rag_analyzer = PDFVectorAnalyzer(PDF_PATH)
            rag_analyzer.load_index(INDEX_PATH, METADATA_PATH)
        
        st.success("Vector index loaded")
        
        analyses = {}
        
        for i, question in enumerate(RESEARCH_QUESTIONS, 1):
            status_text.text(f"Answering question {i}/{len(RESEARCH_QUESTIONS)}...")
            progress_bar.progress(i / len(RESEARCH_QUESTIONS))
            
            answer = rag_analyzer.answer_single_question(
                question,
                top_k=top_k
            )
            
            analyses[question] = answer
        
        progress_bar.empty()
        status_text.empty()
        
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "pdf_path": PDF_PATH,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "top_k": top_k,
                "total_questions": len(RESEARCH_QUESTIONS)
            },
            "qa_pairs": analyses
        }
        
        output_filename = f"research_qa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_filename, "w") as f:
            json.dump(results_data, f, indent=2)
        
        st.success(f"All questions answered. Results saved to {output_filename}")
        
        tab1, tab2, tab3 = st.tabs([
            "All Q&A Pairs",
            "Detailed View",
            "Export Data"
        ])
        
        with tab1:
            st.subheader("Complete Q&A Pairs")
            st.markdown(f"**{len(analyses)} questions answered using RAG**")
            
            for i, (question, answer) in enumerate(analyses.items(), 1):
                with st.expander(f"Q{i}: {question[:100]}...", expanded=False):
                    st.markdown("**Question:**")
                    st.info(question)
                    st.markdown("**Answer (from PDF context):**")
                    st.markdown(answer)
        
        with tab2:
            st.subheader("Detailed Question View")
            
            question_options = [f"Q{i}: {q[:80]}..." for i, q in enumerate(RESEARCH_QUESTIONS, 1)]
            selected_idx = st.selectbox(
                "Select a question to view:",
                range(len(RESEARCH_QUESTIONS)),
                format_func=lambda x: question_options[x]
            )
            
            selected_question = RESEARCH_QUESTIONS[selected_idx]
            selected_answer = analyses[selected_question]
            
            st.markdown("### Question")
            st.info(selected_question)
            
            st.markdown("### Answer")
            st.markdown(selected_answer)
            
            if st.checkbox("Show retrieved PDF chunks"):
                with st.spinner("Retrieving context chunks..."):
                    chunks = rag_analyzer.retrieve_context(selected_question, top_k=top_k)
                    
                    st.markdown(f"**{len(chunks)} relevant chunks retrieved:**")
                    for i, chunk in enumerate(chunks, 1):
                        with st.expander(f"Chunk {i}", expanded=False):
                            st.markdown(chunk)
        
        with tab3:
            st.subheader("Export Q&A Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="Download Complete Results (JSON)",
                    data=json.dumps(results_data, indent=2),
                    file_name=output_filename,
                    mime="application/json",
                    width='stretch'
                )
            
            with col2:
                markdown_content = f"# Health Insurance Market Research Q&A\n\n"
                markdown_content += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                markdown_content += f"**Source:** {PDF_PATH}\n\n"
                markdown_content += "---\n\n"
                
                for i, (question, answer) in enumerate(analyses.items(), 1):
                    markdown_content += f"## Question {i}\n\n"
                    markdown_content += f"**{question}**\n\n"
                    markdown_content += f"{answer}\n\n"
                    markdown_content += "---\n\n"
                
                st.download_button(
                    label="Download as Markdown",
                    data=markdown_content,
                    file_name=f"research_qa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    width='stretch'
                )
            
            st.subheader("Results Preview")
            st.json(results_data, expanded=False)

else:
    st.info("Click 'Answer All Questions with RAG' to start the analysis")
    
    st.markdown("""
    ### How This Works:
    
    **1. Vector Index Creation**
    - PDF is split into overlapping chunks
    - Each chunk is converted to embeddings using OpenAI
    - Embeddings are stored in a FAISS vector database
    
    **2. Question Answering**
    - For each of the 18 questions:
        - Retrieve the most relevant PDF chunks (based on semantic similarity)
        - Send question + context to LLM
        - Generate answer grounded in the PDF content
    
    **3. Pure RAG**
    - No hardcoded answers
    - All responses are generated from actual PDF content
    - Adjust top_k in sidebar to control context size
    
    **Why RAG?**
    - Ensures answers are factual and sourced from the document
    - Reduces hallucination
    - Provides transparent source attribution
    """)

st.divider()
st.caption("Powered by FAISS + OpenAI")