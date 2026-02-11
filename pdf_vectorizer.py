"""
PDF Vectorizer with RAG for Health Insurance Market Research
"""
import os
from pathlib import Path
from typing import List, Dict, Any
import json

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import HumanMessage, SystemMessage

# 18 Research Questions about Health Insurance Markets
RESEARCH_QUESTIONS = [
    "What does the report mean by 'market power' in health insurance, and why is it harmful in both input and output markets?",
    "What is the Herfindahl-Hirschman Index (HHI), and how has market concentration in health insurance changed between 2014 and 2024?",
    "What are the main causes of increased market concentration in health insurance markets over the past decade?",
    "How does vertical integration between insurers and providers affect competition and patient care?",
    "What role do Pharmacy Benefit Managers (PBMs) play in drug pricing, and how does their market power impact costs?",
    "What evidence does the report provide about the relationship between market concentration and premium prices?",
    "How do cross-market mergers affect competition, even when they don't directly overlap geographically?",
    "What are the specific harms to patients and employers from consolidated health insurance markets?",
    "What policy recommendations does the report make to address market concentration in health insurance?",
    "How effective have antitrust enforcement actions been in preventing harmful mergers in health insurance?",
    "What is the role of provider networks in limiting competition, and how do narrow networks affect patient choice?",
    "How does market power in insurance markets differ between individual, small group, and large group markets?",
    "What evidence is provided about information asymmetries and their impact on competition?",
    "How do Medicare Advantage plans contribute to or alleviate market concentration concerns?",
    "What are the barriers to entry for new health insurance competitors, according to the report?",
    "How does the report propose balancing the efficiency gains from scale with the harms from market power?",
    "What specific examples of anticompetitive conduct by major insurers does the report document?",
    "What are the implications of findings in this report for future health insurance regulation and policy?"
]


class PDFVectorAnalyzer:
    """
    Loads, chunks, and indexes a PDF for RAG-based question answering
    """
    
    def __init__(
        self,
        pdf_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        model_name: str = "gpt-4o-mini"
    ):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        
        self.documents = None
        self.chunks = None
        self.vectorstore = None
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model=model_name, temperature=0)
    
    def load_and_split_pdf(self):
        """Load PDF and split into chunks"""
        print(f"Loading PDF: {self.pdf_path}")
        loader = PyPDFLoader(self.pdf_path)
        self.documents = loader.load()
        
        print(f"Loaded {len(self.documents)} pages")
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.chunks = text_splitter.split_documents(self.documents)
        print(f"Created {len(self.chunks)} chunks")
        
        return self.chunks
    
    def build_vector_index(self):
        """Build FAISS vector index from chunks"""
        if not self.chunks:
            raise ValueError("No chunks available. Run load_and_split_pdf() first.")
        
        print("Building vector index...")
        self.vectorstore = FAISS.from_documents(
            self.chunks,
            self.embeddings
        )
        print("Vector index built successfully")
        
        return self.vectorstore
    
    def save_index(self, index_path: str, metadata_path: str):
        """Save vector index and metadata to disk"""
        if not self.vectorstore:
            raise ValueError("No vector store to save. Build index first.")
        
        # Save FAISS index
        self.vectorstore.save_local(index_path)
        
        # Save metadata
        metadata = {
            "pdf_path": self.pdf_path,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "num_chunks": len(self.chunks),
            "num_pages": len(self.documents) if self.documents else 0
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Index saved to {index_path}")
        print(f"Metadata saved to {metadata_path}")
    
    def load_index(self, index_path: str, metadata_path: str):
        """Load vector index and metadata from disk"""
        print(f"Loading index from {index_path}")
        
        self.vectorstore = FAISS.load_local(
            index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"Loaded index with {metadata.get('num_chunks', 'unknown')} chunks")
        
        return self.vectorstore
    
    def retrieve_context(self, question: str, top_k: int = 5) -> List[str]:
        """Retrieve relevant context chunks for a question"""
        if not self.vectorstore:
            raise ValueError("No vector store loaded. Build or load index first.")
        
        # Retrieve relevant documents
        docs = self.vectorstore.similarity_search(question, k=top_k)
        
        # Extract text content
        contexts = [doc.page_content for doc in docs]
        
        return contexts
    
    def answer_single_question(
        self,
        question: str,
        top_k: int = 5
    ) -> str:
        """
        Answer a single question using RAG
        
        Args:
            question: The research question to answer
            top_k: Number of context chunks to retrieve
        
        Returns:
            Generated answer based on PDF context
        """
        # Retrieve relevant context
        contexts = self.retrieve_context(question, top_k=top_k)
        
        # Combine contexts
        combined_context = "\n\n---\n\n".join(contexts)
        
        # Create prompt
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

        # Generate answer
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm(messages)
        
        return response.content
    
    def analyze_all_questions(
        self,
        questions: List[str],
        top_k: int = 5
    ) -> Dict[str, str]:
        """
        Answer all research questions
        
        Args:
            questions: List of research questions
            top_k: Number of context chunks per question
        
        Returns:
            Dictionary mapping questions to answers
        """
        analyses = {}
        
        total = len(questions)
        for i, question in enumerate(questions, 1):
            print(f"Answering question {i}/{total}...")
            answer = self.answer_single_question(question, top_k=top_k)
            analyses[question] = answer
        
        print(f"Completed {total} questions")
        return analyses


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = PDFVectorAnalyzer(
        "competition-health-insurance-us-markets.pdf",
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Load and index PDF
    analyzer.load_and_split_pdf()
    analyzer.build_vector_index()
    analyzer.save_index("faiss_index.bin", "index_metadata.json")
    
    # Answer a single question
    answer = analyzer.answer_single_question(RESEARCH_QUESTIONS[0])
    print(f"\nQuestion: {RESEARCH_QUESTIONS[0]}")
    print(f"\nAnswer: {answer}")