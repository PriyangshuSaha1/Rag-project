# 📚 RAG Project using LangChain, Gemini & Pinecone

This project implements a **Retrieval-Augmented Generation (RAG)** system using:

* 🧠 Google Gemini (LLM + Embeddings)
* 📄 PDF document processing
* ✂️ Text chunking
* 📦 Pinecone vector database
* 🔗 LangChain for orchestration

---

## 🚀 Features

* Load and process PDF documents
* Split documents into chunks
* Convert text into embeddings
* Store embeddings in Pinecone
* Perform semantic search
* Answer user queries based on document context

---

## 📁 Project Structure

```
DAY12/
│── index.js        # Indexing phase (PDF → embeddings → Pinecone)
│── query.js        # Query + chat system
│── NODE.pdf        # Source document
│── package.json
│── .env            # API keys (not uploaded)
│── .gitignore
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/rag-project.git
cd rag-project
```

---

### 2️⃣ Install dependencies

```bash
npm install
```

---

### 3️⃣ Create `.env` file

```env
GEMINI_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
```

---

### 4️⃣ Run Indexing (store data in Pinecone)

```bash
node index.js
```

👉 This will:

* Load PDF
* Create chunks
* Generate embeddings
* Store vectors in Pinecone

---

### 5️⃣ Run Query System

```bash
node query.js
```

👉 Then ask questions like:

```
Ask me anything --> What is Node.js?
```

---

## 🧠 How It Works

### 🔹 Indexing Phase

1. Load PDF using `PDFLoader`
2. Split text into chunks
3. Convert chunks into embeddings (Gemini)
4. Store vectors in Pinecone

---

### 🔹 Query Phase

1. User inputs a question
2. Convert question → embedding
3. Search similar vectors in Pinecone
4. Retrieve top matches (context)
5. Send context + question to Gemini
6. Generate final answer

---

## 📦 Tech Stack

* **LangChain**
* **Google Generative AI (Gemini)**
* **Pinecone**
* **Node.js**

---

## ⚠️ Important Notes

* ❌ Do NOT upload:

  * `node_modules/`
  * `.env`
* ✅ Always include:

  * `package.json`
  * source code

---

## 💡 Future Improvements

* Add UI (React / Next.js)
* Streaming responses
* Multi-document support
* Better prompt engineering
* Intent detection model

---

## 👨‍💻 Author

Your Name
GitHub: https://PriyangshuSaha1

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub!
