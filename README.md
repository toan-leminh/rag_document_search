# rag_document_search
Use LLM, RAG to search on document (PDF,... )

-Language: Python3 (>=3.9)
-Framwork: Langchain 
-Embedding Model: OpenAI Embedding (text-embedding-3-small)
-LLM: GPT-4-turbo
-Vector DB: FAISS
-Frontend: Streamlit


# Environment
python -m venv venv
source venv/bin/activate
pip install python-dotenv  langchain-openai langchain-community langchain openai faiss-cpu streamlit pypdf

# Run 
python main.py