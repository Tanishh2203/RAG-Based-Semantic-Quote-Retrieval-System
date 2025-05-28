# RAG-Based Semantic Quote Retrieval System

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system for semantic quote retrieval using the Abirate/english_quotes dataset. It includes data preprocessing, model fine-tuning, RAG pipeline, evaluation, and a Streamlit web application.

## Setup Instructions

1. **Environment Setup**:

   - Install Python.
   - Create a virtual environment: `python -m venv venv`
   - Activate it: `.\venv\Scripts\activate` (Windows)
   - Install dependencies: `pip install datasets transformers sentence-transformers faiss-cpu streamlit pandas numpy ragas langchain langchain-community torch ollama`

2. **Ollama (Optional)**:

   - Install Ollama from [ollama.ai](https://ollama.ai/).
   - Pull Llama2 model: `ollama pull llama2`
   - Start Ollama server: `ollama serve`

3. **Running the Notebook**:

   - Open `rag.ipynb` in VS Code.
   - Select the virtual environment kernel.
   - Run cells sequentially to load data, initialize the RAG pipeline, and evaluate.

4. **Running the Streamlit App**:
   - Save `rag_system.py` and `app.py`.
   - Run: `streamlit run app.py`
   - Access the app at `http://localhost:8501`.

## Files

- `rag_quote_retrieval.ipynb`: ipynb file with all code.
- `rag_system.py`: Core classes for data processing, model, and RAG pipeline.
- `app.py`: Streamlit application.
- `fine_tuned_quote_model/`: Fine-tuned model directory.'
- `top-tags.png`
- `quote_length_distribution.png`

## Design Choices

- **Model**: Used `all-MiniLM-L6-v2` for efficient embedding generation.
- **Retrieval**: FAISS with cosine similarity for fast quote retrieval.
- **LLM**: Supports Ollama with Llama2 or fallback response generator.
- **Evaluation**: Uses RAGAS for metrics, with manual evaluation as fallback.

## Challenges

- Fine-tuning can be resource-intensive; limited sample size to 1000 for training.
- Ollama setup requires additional configuration; fallback ensures functionality.
- RAGAS evaluation may require API keys for some metrics.

## How to Run

- Follow the notebook cells for testing and evaluation.
- Use the Streamlit app for interactive query processing.
- Example queries:
  - "Quotes about insanity attributed to Einstein"
  - "Motivational quotes tagged 'accomplishment'"
  - "All Oscar Wilde quotes with humor"
