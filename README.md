# RAG-based Document QA System

A Retrieval-Augmented Generation (RAG) system built with **Python, Gemini LLM, LangChain, and Pinecone**. This application allows users to upload PDF documents, processes them into semantic vector embeddings, and provides an interactive chat interface to answer questions based strictly on the uploaded context.

## Features
- **PDF Processing**: Seamlessly upload and chunk large PDF documents using `RecursiveCharacterTextSplitter`.
- **Vector Embeddings**: Generates highly accurate semantic embeddings using Google's `gemini-embedding-001`.
- **Vector Database**: Stores and retrieves embeddings instantly using `Pinecone` for sub-second query latency.
- **Conversational AI**: Uses Google's `gemini-2.5-flash` model to answer queries based on the top semantic matches, reducing hallucinations.
- **Interactive UI**: Built with `Streamlit` for a clean, responsive, and easy-to-use web interface.

## Tech Stack
- **Language**: Python
- **LLM / GenAI**: Google Gemini (Flash & Embeddings)
- **Framework**: LangChain, Streamlit
- **Vector Database**: Pinecone

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/PriyangshuSaha1/Rag-project.git
   cd Rag-project
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Environment Variables**:
   Create a `.env` file in the root directory and add your API keys (or you can input them directly via the Streamlit UI):
   ```env
   GEMINI_API_KEY=your_google_gemini_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX_NAME=your_pinecone_index_name
   ```

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Open the app in your browser (usually `http://localhost:8501`).
2. Provide your API keys in the sidebar (if not using a `.env` file).
3. Upload a PDF document and click **"Process & Index Document"**.
4. Ask questions in the chat interface! The system will retrieve the top 5 most relevant chunks from the document and use the Gemini LLM to construct an accurate answer.
