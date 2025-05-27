# Trinity v0.1 

Trinity is a lightweight Streamlit-based PDF QA assistant that uses powerful language models (via Groq) and local embedding models (via Ollama) to let you chat with your documents.

##  Features

- Upload multiple PDFs and ask questions directly from them.
- Uses FAISS for fast document retrieval.
- Powered by `deepseek-r1-distill-llama-70b` via the Groq API.
- Local embedding model: `nomic-embed-text:v1.5` via Ollama.
- Fast and interactive chat UI using Streamlit.

##  Requirements

- Python 3.8+
- [Ollama](https://ollama.com/) installed and running (for embeddings).
- Groq API key.
- Create an .env file for storing your API keys.For Groq, do "GROQ_API_KEY=your_groq_api_key_here"

##  Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/trinity-pdf-qa.git
   cd trinity-pdf-qa
