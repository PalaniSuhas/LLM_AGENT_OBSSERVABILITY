"""
PDF Vectorization and RAG-based Analysis System
Uses FAISS for vector search to avoid caching issues
"""
import os
import json
from typing import List, Dict, Any
from pathlib import Path

# PDF processing
from PyPDF2 import PdfReader

# Vector store
import faiss
import numpy as np

# Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

# For structured document loading
from langchain.docstore.document import Document


class PDFVectorAnalyzer:
    """
    Vectorizes PDF and uses RAG to answer questions without caching.
    Each query retrieves fresh context from the vector store.
    """
    
    def __init__(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embeddings model
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize LLM for analysis
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,  # Some randomness to avoid exact caching
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Storage for chunks and index
        self.documents: List[Document] = []
        self.index = None
        self.chunk_texts: List[str] = []
        
    def load_and_split_pdf(self) -> List[Document]:
        """Load PDF and split into chunks."""
        print(f"Loading PDF from {self.pdf_path}...")
        
        # Read PDF
        reader = PdfReader(self.pdf_path)
        
        # Extract text from all pages
        full_text = ""
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            full_text += f"\n\n--- Page {page_num + 1} ---\n\n{text}"
        
        print(f"Extracted {len(full_text)} characters from PDF")
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_text(full_text)
        
        # Create Document objects
        self.documents = [
            Document(page_content=chunk, metadata={"chunk_id": i})
            for i, chunk in enumerate(chunks)
        ]
        
        print(f"Split into {len(self.documents)} chunks")
        return self.documents
    
    def build_vector_index(self):
        """Build FAISS index from document chunks."""
        print("Building FAISS vector index...")
        
        if not self.documents:
            raise ValueError("No documents loaded. Call load_and_split_pdf() first.")
        
        # Extract texts
        self.chunk_texts = [doc.page_content for doc in self.documents]
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings_list = self.embeddings.embed_documents(self.chunk_texts)
        embeddings_array = np.array(embeddings_list, dtype=np.float32)
        
        # Create FAISS index
        dimension = embeddings_array.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_array)
        
        print(f"FAISS index built with {self.index.ntotal} vectors")
    
    def retrieve_context(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve top-k most relevant chunks for a query."""
        if self.index is None:
            raise ValueError("Index not built. Call build_vector_index() first.")
        
        # Embed the query
        query_embedding = self.embeddings.embed_query(query)
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # Search FAISS index
        distances, indices = self.index.search(query_vector, top_k)
        
        # Return relevant chunks
        relevant_chunks = [self.chunk_texts[idx] for idx in indices[0]]
        return relevant_chunks
    
    def analyze_with_rag(
        self,
        question: str,
        context_chunks: List[str],
        benchmark_results: Dict[str, Any]
    ) -> str:
        """
        Use RAG to analyze benchmark results with PDF context.
        Each call retrieves fresh context to avoid caching.
        """
        
        # Combine context
        context = "\n\n---\n\n".join(context_chunks)
        
        # Prepare benchmark data summary
        benchmark_summary = self._summarize_benchmark_data(benchmark_results)
        
        # Create prompt with fresh context
        prompt = f"""You are analyzing LLM observability tool benchmarks in the context of health insurance market research.

HEALTH INSURANCE MARKET CONTEXT (from research PDF):
{context}

BENCHMARK RESULTS:
{benchmark_summary}

RESEARCH QUESTION:
{question}

YOUR TASK:
1. Analyze how the benchmark results relate to the research question
2. Use insights from the health insurance market context to inform your analysis
3. Provide specific recommendations based on both the benchmark data and market research principles
4. Be concrete and data-driven in your conclusions

Provide a comprehensive analysis:"""

        # Get fresh response (temperature > 0 reduces caching)
        response = self.llm.invoke([{"role": "user", "content": prompt}])
        
        return response.content
    
    def _summarize_benchmark_data(self, results: Dict[str, Any]) -> str:
        """Summarize benchmark results for the prompt."""
        summary_lines = []
        
        for tool_name, runs in results.items():
            if not runs or "error" in runs[0]:
                continue
            
            valid_runs = [r for r in runs if "error" not in r]
            if not valid_runs:
                continue
            
            avg_latency = sum(r["latency_ms"] for r in valid_runs) / len(valid_runs)
            avg_total = sum(r["total_time"] for r in valid_runs) / len(valid_runs)
            
            summary_lines.append(
                f"- {tool_name}: {len(valid_runs)} runs, "
                f"avg latency {avg_latency:.2f}ms, "
                f"avg total time {avg_total:.2f}ms, "
                f"tracing: {valid_runs[0]['metadata'].get('tracing', 'unknown')}"
            )
        
        return "\n".join(summary_lines)
    
    def analyze_all_questions(
        self,
        questions: List[str],
        benchmark_results: Dict[str, Any],
        top_k: int = 5
    ) -> Dict[str, str]:
        """
        Analyze all questions using RAG.
        Returns a dictionary mapping questions to analysis.
        """
        analyses = {}
        
        for i, question in enumerate(questions, 1):
            print(f"\nAnalyzing question {i}/{len(questions)}: {question[:80]}...")
            
            # Retrieve fresh context for this question
            context_chunks = self.retrieve_context(question, top_k=top_k)
            
            # Analyze with RAG
            analysis = self.analyze_with_rag(
                question=question,
                context_chunks=context_chunks,
                benchmark_results=benchmark_results
            )
            
            analyses[question] = analysis
        
        return analyses
    
    def save_index(self, index_path: str, metadata_path: str):
        """Save FAISS index and metadata to disk."""
        if self.index is None:
            raise ValueError("No index to save")
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        metadata = {
            "chunk_texts": self.chunk_texts,
            "num_chunks": len(self.chunk_texts),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Index saved to {index_path}")
        print(f"Metadata saved to {metadata_path}")
    
    def load_index(self, index_path: str, metadata_path: str):
        """Load FAISS index and metadata from disk."""
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.chunk_texts = metadata["chunk_texts"]
        self.chunk_size = metadata["chunk_size"]
        self.chunk_overlap = metadata["chunk_overlap"]
        
        print(f"Index loaded from {index_path}")
        print(f"Loaded {len(self.chunk_texts)} chunks")


# 18 Research Questions
RESEARCH_QUESTIONS = [
    # Conceptual & Understanding
    "What does the report mean by 'market power' in health insurance, and why is it harmful in both input and output markets?",
    "How do health insurers function as intermediaries between providers and consumers?",
    "What is the Herfindahl-Hirschman Index (HHI), and why is it used to measure market concentration?",
    "According to DOJ/FTC Merger Guidelines, what threshold defines a 'highly concentrated' market?",
    
    # Data & Methodology
    "Why does the study analyze health insurance markets at the Metropolitan Statistical Area (MSA) level rather than only at the national level?",
    "What data sources are used in this study, and why is Decision Resources Group considered reliable?",
    "Why are PPO, HMO, POS, Exchanges, and Medicare Advantage treated as separate or combined product markets?",
    "What methodological steps are taken to ensure enrollment data accurately reflects the insured population?",
    
    # Key Findings
    "What evidence does the study provide to show that most U.S. commercial health insurance markets are highly concentrated?",
    "How has market concentration in commercial insurance markets changed between 2014 and 2024?",
    "What role did public health insurance exchanges play in moderating overall market concentration trends?",
    "How do Medicare Advantage market concentration trends compare to commercial market trends?",
    
    # Competition & Market Dynamics
    "How does insurer consolidation affect insurance premiums and consumer choice?",
    "What is monopsony power, and how can health insurers exercise it over physicians?",
    "Why does physician practice size matter when evaluating insurer bargaining power?",
    
    # Antitrust & Policy Implications
    "Why do proposed mergers among health insurers raise antitrust concerns, according to the report?",
    "What lessons can be drawn from past blocked mergers such as Anthem–Cigna or Aetna–Prudential?",
    "Based on the findings, what policy actions should regulators consider to promote competition in health insurance markets?"
]


if __name__ == "__main__":
    # Example usage
    PDF_PATH = "competition-health-insurance-us-markets.pdf"
    
    # Initialize analyzer
    analyzer = PDFVectorAnalyzer(PDF_PATH)
    
    # Load and process PDF
    analyzer.load_and_split_pdf()
    analyzer.build_vector_index()
    
    # Save index for reuse
    analyzer.save_index("faiss_index.bin", "index_metadata.json")
    
    # Example: Test with a sample question
    sample_question = RESEARCH_QUESTIONS[0]
    context = analyzer.retrieve_context(sample_question, top_k=3)
    
    print("\n" + "="*80)
    print(f"SAMPLE QUESTION: {sample_question}")
    print("="*80)
    print("\nRETRIEVED CONTEXT:")
    for i, chunk in enumerate(context, 1):
        print(f"\n--- Chunk {i} ---")
        print(chunk[:300] + "...")