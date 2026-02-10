

## Installation and Run Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Create `.env` file

In the project root directory:

```bash
# OpenAI
OPENAI_API_KEY=sk-your-openai-key

# LangSmith
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_PROJECT=agent-benchmark

# Langfuse
LANGFUSE_SECRET_KEY=your-langfuse-secret
LANGFUSE_PUBLIC_KEY=your-langfuse-public
LANGFUSE_HOST=https://us.cloud.langfuse.com

# Braintrust
BRAINTRUST_API_KEY=your-braintrust-key

```



---

### 3. Download the PDF

Download the PDF from this link:

[https://www.ama-assn.org/system/files/competition-health-insurance-us-markets.pdf](https://www.ama-assn.org/system/files/competition-health-insurance-us-markets.pdf)

Save it in the project root directory as:

```
competition-health-insurance-us-markets.pdf
```

---

### 4. Build the vector index (run first)

```bash
python build_index.py
```

This creates:

* `faiss_index.bin`
* `index_metadata.json`

---

### 5. Test the RAG system (run second)

```bash
python test_rag_system.py
```

This validates embeddings, FAISS, and API access.

---

### 6. Run the Streamlit benchmark app

```bash
python -m streamlit run benchmark_rag.py
```

---

## Required Run Order

```
pip install -r requirements_rag.txt
python build_index.py
python test_rag_system.py
python -m streamlit run benchmark_rag.py
```

