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

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from google import genai
from groq import Groq

load_dotenv()

st.set_page_config(page_title="RAG Intelligence", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .stChatMessage { border-radius: 10px; padding: 10px; margin-bottom: 10px; }
    .main-title { font-family: 'Inter', sans-serif; color: #ffffff; font-size: 2.5rem; font-weight: 700; margin-bottom: 0px; }
    .sub-title { color: #a0aec0; font-size: 1.1rem; margin-bottom: 30px; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    [data-testid="stStatusWidget"] {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🧠 RAG Intelligence Studio</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Securely query your documents using Local Embeddings & Pinecone</p>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am ready to answer questions based on your indexed documents. What would you like to know?"}]

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

@st.cache_resource(show_spinner=False)
def get_llm(provider, api_key):
    if provider == "Google Gemini":
        try:
            # AUTO-DISCOVERY: Fetch allowed models
            client = genai.Client(api_key=api_key)
            allowed_models = [m.name for m in client.models.list()]
            
            best_model = None
            for pref in ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash", "gemini-1.0-pro", "gemini-pro"]:
                if any(pref in m for m in allowed_models):
                    best_model = [m for m in allowed_models if pref in m][0]
                    break
                    
            if not best_model and allowed_models:
                best_model = allowed_models[0]
                
            if not best_model:
                raise ValueError("Your Google API Key does not have access to any models.")
                
            best_model_name = best_model.replace("models/", "")
            return ChatGoogleGenerativeAI(model=best_model_name, google_api_key=api_key, temperature=0.3), best_model_name
        except Exception as e:
            raise Exception(f"Google API Key verification failed: {e}")
    else:
        try:
            # GROQ AUTO-DISCOVERY: Fetch exactly what Groq currently supports!
            client = Groq(api_key=api_key)
            models = client.models.list().data
            # Filter out whisper (audio) and get the first text model
            text_models = [m.id for m in models if "whisper" not in m.id]
            
            if not text_models:
                raise ValueError("Your Groq API Key has no text models available.")
                
            best_model_name = text_models[0]
            
            # Prefer a fast llama model if available
            for pref in ["llama", "gemma", "mixtral"]:
                for m in text_models:
                    if pref in m:
                        best_model_name = m
                        break
                if "llama" in best_model_name:
                    break
                    
            return ChatGroq(model=best_model_name, groq_api_key=api_key, temperature=0.3), best_model_name
        except Exception as e:
            raise Exception(f"Groq API Key verification failed: {e}")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4233/4233830.png", width=60)
    st.header("⚙️ Configuration")
    
    with st.expander("🔑 API Credentials", expanded=True):
        llm_provider = st.selectbox("AI Model Provider", ["Google Gemini", "Groq (Llama 3)"])
        
        if llm_provider == "Google Gemini":
            api_key = st.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
        else:
            api_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
            
        pinecone_api_key = st.text_input("Pinecone API Key", type="password", value=os.getenv("PINECONE_API_KEY", ""))
        pinecone_index   = st.text_input("Pinecone Index Name", value=os.getenv("PINECONE_INDEX_NAME", ""))

    st.divider()
    st.header("📄 Knowledge Base")
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])

    if st.button("🚀 Process & Index Document", use_container_width=True):
        if not (api_key and pinecone_api_key and pinecone_index):
            st.error("Please provide all credentials above.")
        elif uploaded_file is not None:
            
            with st.status("🚀 Starting Document Processing...", expanded=True) as status:
                try:
                    os.environ["PINECONE_API_KEY"] = pinecone_api_key
                    st.write("⏳ Downloading / Loading local AI embeddings...")
                    
                    embeddings = get_embeddings()
                    st.write("✅ Embeddings loaded successfully!")
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    st.write("⏳ Reading PDF and creating text chunks...")
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()

                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    splits = splitter.split_documents(docs)
                    st.write(f"✅ Document successfully split into {len(splits)} chunks!")
                    
                    st.write("⏳ Generating vectors and uploading to Pinecone Database...")
                    PineconeVectorStore.from_documents(splits, embeddings, index_name=pinecone_index)
                    os.remove(tmp_path)
                    st.write("✅ Vectors successfully stored in Pinecone!")
                    
                    status.update(label="🎉 Document Processed & Indexed Successfully!", state="complete", expanded=False)
                    
                except Exception as e:
                    status.update(label="❌ Error during processing", state="error", expanded=True)
                    st.error(f"Error: {e}")
        else:
            st.warning("Please upload a PDF first.")
            
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": "History cleared. How can I help?"}]

if not (api_key and pinecone_api_key and pinecone_index):
    st.info("👈 Please enter your API credentials in the sidebar to start chatting.")
    st.stop()

os.environ["PINECONE_API_KEY"] = pinecone_api_key

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("📑 View Source Context"):
                for i, doc in enumerate(msg["sources"]):
                    st.markdown(f"**Chunk {i+1} (Page {doc.metadata.get('page', '?')})**")
                    st.info(doc.page_content)

if prompt_text := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    with st.chat_message("assistant"):
        with st.spinner(f"Analyzing..."):
            try:
                embeddings = get_embeddings()
                llm, actual_model_name = get_llm(llm_provider, api_key)
                
                vectorstore = PineconeVectorStore(index_name=pinecone_index, embedding=embeddings)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

                prompt_template = ChatPromptTemplate.from_template("""
You are a helpful AI assistant answering questions based on the provided documentation.

Context from the documentation:
{context}

Question: {question}

Instructions:
- Answer using ONLY the information from the context above.
- If the answer is not in the context, clearly state that you don't have enough information.
- Use Markdown formatting for readability.

Answer:""")

                def format_docs(docs):
                    return "\n\n---\n\n".join(doc.page_content for doc in docs)

                rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt_template
                    | llm
                    | StrOutputParser()
                )

                answer = rag_chain.invoke(prompt_text)
                source_docs = retriever.invoke(prompt_text)
                
                st.markdown(answer)
                st.caption(f"✨ Generated using {actual_model_name}")
                
                with st.expander("📑 View Source Context"):
                    for i, doc in enumerate(source_docs):
                        st.markdown(f"**Chunk {i+1} (Page {doc.metadata.get('page', '?')})**")
                        st.info(doc.page_content)
                        
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": source_docs
                })

            except Exception as e:
                error_msg = f"An error occurred: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
