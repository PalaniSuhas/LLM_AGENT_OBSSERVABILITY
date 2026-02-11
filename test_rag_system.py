#!/usr/bin/env python3
"""
Test script to verify RAG system functionality
Run this to test the complete pipeline before using the Streamlit app
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def test_environment():
    """Test that all required environment variables are set"""
    print("\n" + "="*80)
    print("TESTING ENVIRONMENT")
    print("="*80)
    
    required = ["OPENAI_API_KEY"]
    optional = ["LANGCHAIN_API_KEY", "LANGFUSE_PUBLIC_KEY", "BRAINTRUST_API_KEY"]
    
    print("\n Required environment variables:")
    for var in required:
        value = os.getenv(var)
        if value:
            print(f"  [PASS] {var}: {'*' * 8}{value[-4:]}")
        else:
            print(f"  [FAIL] {var}: NOT SET")
            return False
    
    print("\n Optional environment variables (for full testing):")
    for var in optional:
        value = os.getenv(var)
        if value:
            print(f"  [PASS] {var}: {'*' * 8}{value[-4:]}")
        else:
            print(f"  [SKIP] {var}: not set (optional)")
    
    return True

def test_dependencies():
    """Test that all required packages are installed"""
    print("\n" + "="*80)
    print("TESTING DEPENDENCIES")
    print("="*80)
    
    required_packages = [
        ("faiss", "faiss-cpu"),
        ("PyPDF2", "PyPDF2"),
        ("langchain", "langchain"),
        ("langchain_openai", "langchain-openai"),
        ("streamlit", "streamlit"),
        ("pandas", "pandas"),
        ("plotly", "plotly"),
    ]
    
    all_installed = True
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            print(f"  [PASS] {package_name}")
        except ImportError:
            print(f"  [FAIL] {package_name} - NOT INSTALLED")
            all_installed = False
    
    return all_installed

def test_pdf_exists():
    """Test that the PDF file exists"""
    print("\n" + "="*80)
    print("TESTING PDF FILE")
    print("="*80)
    
    pdf_path = "competition-health-insurance-us-markets.pdf"
    
    if Path(pdf_path).exists():
        size = Path(pdf_path).stat().st_size
        print(f"  [PASS] PDF found: {pdf_path}")
        print(f"  [INFO] File size: {size:,} bytes ({size/1024/1024:.2f} MB)")
        return True
    else:
        print(f"  [FAIL] PDF not found: {pdf_path}")
        print(f"  Please download and place the PDF in the current directory")
        return False

def test_vector_index():
    """Test that vector index exists or can be created"""
    print("\n" + "="*80)
    print("TESTING VECTOR INDEX")
    print("="*80)
    
    index_path = "faiss_index.bin"
    metadata_path = "index_metadata.json"
    
    if Path(index_path).exists() and Path(metadata_path).exists():
        print(f"  [PASS] Vector index found: {index_path}")
        print(f"  [PASS] Metadata found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"  [INFO] Index contains {metadata['num_chunks']} chunks")
        print(f"  [INFO] Chunk size: {metadata['chunk_size']} characters")
        print(f"  [INFO] Chunk overlap: {metadata['chunk_overlap']} characters")
        return True
    else:
        print(f"  [SKIP] Vector index not found")
        print(f"  Run: python build_index.py")
        return False

def test_rag_retrieval():
    """Test that RAG retrieval works"""
    print("\n" + "="*80)
    print("TESTING RAG RETRIEVAL")
    print("="*80)
    
    try:
        from pdf_vectorizer import PDFVectorAnalyzer, RESEARCH_QUESTIONS
        
        if not Path("faiss_index.bin").exists():
            print("  [SKIP] Skipping (index not built yet)")
            return True
        
        print("  Loading vector index...")
        analyzer = PDFVectorAnalyzer("competition-health-insurance-us-markets.pdf")
        analyzer.load_index("faiss_index.bin", "index_metadata.json")
        
        test_question = RESEARCH_QUESTIONS[0]
        print(f"  Testing with: {test_question[:60]}...")
        
        chunks = analyzer.retrieve_context(test_question, top_k=3)
        print(f"  [PASS] Retrieved {len(chunks)} relevant chunks")
        
        if chunks:
            preview = chunks[0][:150].replace('\n', ' ')
            print(f"  [INFO] Sample chunk: {preview}...")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("RAG-BASED BENCHMARK SYSTEM - TEST SUITE")
    print("="*80)
    
    results = []
    
    results.append(("Environment", test_environment()))
    results.append(("Dependencies", test_dependencies()))
    results.append(("PDF File", test_pdf_exists()))
    results.append(("Vector Index", test_vector_index()))
    results.append(("RAG Retrieval", test_rag_retrieval()))
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status}: {test_name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] ALL TESTS PASSED")
        print("\nYou can now:")
        print("  1. Build index: python build_index.py")
        print("  2. Run benchmark: streamlit run benchmark_rag.py")
    else:
        print("\n[ERROR] SOME TESTS FAILED")
        print("\nPlease fix the issues above before proceeding.")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    main()