# Enterprise Knowledge Assistant

A Streamlit RAG assistant for searching and summarizing company knowledge across PDFs, Word documents, and Excel workbooks.

## What It Does

- Ingests `.pdf`, `.docx`, `.xlsx`, `.xls`, and `.csv` files
- Splits documents into searchable chunks
- Builds a local FAISS vector index with Hugging Face embeddings
- Answers employee questions with citations
- Summarizes uploaded document collections
- Uses Llama 3 through Hugging Face Inference Endpoints

## Tech Stack

- Streamlit
- LangChain
- FAISS
- Hugging Face embeddings
- Llama 3

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set your Hugging Face token:

```env
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

Llama 3 models on Hugging Face may require license acceptance for your account.

## Run

```powershell
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Usage

1. Upload company documents in the sidebar.
2. Click **Build Knowledge Base**.
3. Ask a question or request a summary.
4. Review the answer and citations.

## Notes

- FAISS indexes are created in memory for the current Streamlit session.
- Uploaded files are temporarily saved under `.streamlit_uploads/`.
- The default embedding model is `sentence-transformers/all-MiniLM-L6-v2`.
- The default LLM endpoint model is `meta-llama/Meta-Llama-3-8B-Instruct`.

## Deployment

- Local Streamlit app: `streamlit run app.py`
- Vercel lightweight API endpoint: `https://enterprise-knowledge-assistant-i0xhnscvl.vercel.app`
- The Vercel deployment serves a simple status endpoint; the main Streamlit interface is intended to run locally.
