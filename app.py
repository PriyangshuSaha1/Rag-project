import streamlit as st
import os
import tempfile
from typing import List, Optional, Any
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from google import genai

load_dotenv()

# ─── Custom Embeddings (google-genai SDK) ──────────────────────────────────────
class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "gemini-embedding-001", output_dimensionality: int = 768):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.output_dimensionality = output_dimensionality

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = []
        for text in texts:
            r = self.client.models.embed_content(
                model=self.model,
                contents=text,
                config={"output_dimensionality": self.output_dimensionality}
            )
            result.append(r.embeddings[0].values)
        return result

    def embed_query(self, text: str) -> List[float]:
        r = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config={"output_dimensionality": self.output_dimensionality}
        )
        return r.embeddings[0].values


# ─── Custom LLM (new Interactions API) ────────────────────────────────────────
class GeminiLLM(LLM):
    api_key: str
    model: str = "gemini-3.6-flash"
    temperature: float = 0.3

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        client = genai.Client(api_key=self.api_key)
        # Use the new Interactions API as recommended by Google
        response = client.interactions.create(
            model=self.model,
            input=prompt
        )
        # Extract text from ModelOutputStep inside steps
        for step in (response.steps or []):
            if hasattr(step, 'content') and step.content:
                for content in step.content:
                    if hasattr(content, 'text') and content.text:
                        return content.text
        return ""


# ─── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Document QA System", layout="wide")
st.title("📄 RAG-based Document QA System")
st.markdown("Built with **Python, Gemini LLM, LangChain, and Pinecone**")

st.sidebar.header("⚙️ Configuration")
gemini_api_key   = st.sidebar.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
pinecone_api_key = st.sidebar.text_input("Pinecone API Key", type="password", value=os.getenv("PINECONE_API_KEY", ""))
pinecone_index   = st.sidebar.text_input("Pinecone Index Name", value=os.getenv("PINECONE_INDEX_NAME", ""))

if not (gemini_api_key and pinecone_api_key and pinecone_index):
    st.warning("⚠️ Please provide all three credentials in the sidebar to proceed.")
    st.stop()

os.environ["PINECONE_API_KEY"] = pinecone_api_key

try:
    embeddings = GeminiEmbeddings(api_key=gemini_api_key, model="gemini-embedding-001", output_dimensionality=768)
    llm = GeminiLLM(api_key=gemini_api_key, model="gemini-3.6-flash", temperature=0.3)
except Exception as e:
    st.error(f"Error initializing models: {e}")
    st.stop()

# ─── Document Upload & Indexing ────────────────────────────────────────────────
st.sidebar.header("📁 Document Upload")
uploaded_file = st.sidebar.file_uploader("Upload a PDF document to index", type=["pdf"])

if st.sidebar.button("Process & Index Document"):
    if uploaded_file is not None:
        with st.spinner("Processing and chunking document..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                loader = PyPDFLoader(tmp_path)
                docs = loader.load()

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = splitter.split_documents(docs)

                st.sidebar.info(f"✂️ Created {len(splits)} chunks. Uploading to Pinecone...")

                PineconeVectorStore.from_documents(splits, embeddings, index_name=pinecone_index)

                os.remove(tmp_path)
                st.sidebar.success("✅ Document indexed in Pinecone successfully!")

            except Exception as e:
                st.sidebar.error(f"An error occurred: {e}")
    else:
        st.sidebar.error("Please upload a PDF first.")

# ─── Chat / QA Interface ───────────────────────────────────────────────────────
st.header("💬 Ask Questions")
user_question = st.text_input("Enter your question based on the indexed document:")

if st.button("Get Answer"):
    if user_question:
        with st.spinner("Searching vector database and generating answer..."):
            try:
                vectorstore = PineconeVectorStore(index_name=pinecone_index, embedding=embeddings)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

                prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant answering questions based on the provided documentation.

Context from the documentation:
{context}

Question: {question}

Instructions:
- Answer using ONLY the information from the context above.
- If the answer is not in the context, say "I don't have enough information in the provided document to answer that."
- Be concise, clear, and use bullet points if appropriate.

Answer:""")

                def format_docs(docs):
                    return "\n\n---\n\n".join(doc.page_content for doc in docs)

                rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                answer = rag_chain.invoke(user_question)

                st.subheader("✅ Answer:")
                st.success(answer)

                with st.expander("🔍 View Source Document Chunks"):
                    source_docs = retriever.invoke(user_question)
                    for i, doc in enumerate(source_docs):
                        st.write(f"**Chunk {i+1} (Page {doc.metadata.get('page', 'Unknown')}):**")
                        st.info(doc.page_content)

            except Exception as e:
                st.error(f"An error occurred during retrieval: {e}")
    else:
        st.warning("Please enter a question.")
