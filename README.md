# Simple-RAG

Simple-RAG is a small retrieval-augmented generation app that reads PDF files from `dataset/`, builds a FAISS vector store, and answers a user question with a NVIDIA-hosted LLM. The project also includes `md.py`, a helper script that converts the PDFs in `dataset/` into Markdown files.

## Technologies Used

- `Python`
- `LangChain`
- `FAISS`
- `langchain-nvidia-ai-endpoints`
- `python-dotenv`
- `pypdf`
- `unstructured[pdf]`

## Installation

1. Create a virtual environment:

```bash
python -m venv .venv
```

2. Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file in the project root. The chatbot requires `NVIDIA_API_KEY`.

```env
NVIDIA_API_KEY=your_api_key_here
```

The repository also includes `.env.example` with the sample environment variables used in the project.

## Run the Application

Place one or more PDF files in `dataset/`, then run the chatbot:

```bash
python chatbot.py
```

The script will ask for a question, generate an answer, and save the response to `answer.txt`.

To convert the PDFs in `dataset/` to Markdown files, run:

```bash
python md.py
```
