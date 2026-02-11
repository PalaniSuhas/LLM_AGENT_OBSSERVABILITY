#!/usr/bin/env python3
"""
Standalone script to build FAISS vector index from PDF
Run this once to prepare the index before using the research Q&A tool
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from pdf_vectorizer import PDFVectorAnalyzer, RESEARCH_QUESTIONS

load_dotenv()

def main():
    # Configuration
    PDF_PATH = "competition-health-insurance-us-markets.pdf"
    INDEX_PATH = "faiss_index.bin"
    METADATA_PATH = "index_metadata.json"
    
    # Check if PDF exists
    if not Path(PDF_PATH).exists():
        print(f"Error: PDF not found at {PDF_PATH}")
        print(f"Please place the PDF in the current directory")
        return
    
    print("="*80)
    print("BUILDING FAISS VECTOR INDEX")
    print("="*80)
    
    # Initialize analyzer
    print(f"\nSource PDF: {PDF_PATH}")
    analyzer = PDFVectorAnalyzer(
        PDF_PATH,
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Load and process PDF
    print("\nStep 1: Loading and splitting PDF...")
    docs = analyzer.load_and_split_pdf()
    print(f"   Created {len(docs)} document chunks")
    
    # Build FAISS index
    print("\nStep 2: Building FAISS vector index...")
    analyzer.build_vector_index()
    
    # Get the number of indexed vectors from the vectorstore
    if analyzer.vectorstore:
        num_vectors = analyzer.vectorstore.index.ntotal
        print(f"   Indexed {num_vectors} vectors")
    
    # Save index
    print("\nStep 3: Saving index to disk...")
    analyzer.save_index(INDEX_PATH, METADATA_PATH)
    
    # The index is saved to a directory (without .bin extension)
    index_dir = INDEX_PATH.replace('.bin', '')
    print(f"   Index saved to directory: {index_dir}/")
    print(f"   Metadata saved to: {METADATA_PATH}")
    
    # Test retrieval with a sample question
    print("\nStep 4: Testing retrieval with sample question...")
    sample_question = RESEARCH_QUESTIONS[0]
    print(f"\n   Question: {sample_question}")
    
    context_chunks = analyzer.retrieve_context(sample_question, top_k=3)
    print(f"\n   Retrieved {len(context_chunks)} relevant chunks:")
    
    for i, chunk in enumerate(context_chunks, 1):
        print(f"\n   --- Chunk {i} ---")
        # Show first 200 characters
        preview = chunk[:200].replace('\n', ' ')
        print(f"   {preview}...")
    
    print("\n" + "="*80)
    print("INDEX BUILD COMPLETE")
    print("="*80)
    print(f"\nIndex directory created: {index_dir}/")
    print(f"\nYou can now run the Q&A tool:")
    print(f"  streamlit run research_qa.py")
    print()

if __name__ == "__main__":
    main()