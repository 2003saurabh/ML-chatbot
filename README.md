
# 🚀 RAG-SELF: Modular RAG Application with Admin & User Interfaces

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red)](https://streamlit.io/)
[![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-purple)](https://qdrant.tech/)
[![AWS](https://img.shields.io/badge/Cloud-AWS-orange)](https://aws.amazon.com/)

---

## 📄 Overview

**RAG-SELF** is a **production-grade Retrieval-Augmented Generation (RAG)** system with a **clear separation of Admin and User modules**.  
Admins can upload PDFs, generate embeddings, and manage vector storage. Users can query the system using an LLM with **contextual memory** and receive intelligent, document-grounded responses.

The app is cloud-ready and uses:
- **AWS S3** for document storage
- **Amazon Bedrock** (Titan for Embeddings, Llama 3 for LLM)
- **Qdrant** for vector search
- **Streamlit** for UI
- **LangChain** for LLM orchestration

---

## ✨ Key Features

### Admin Portal
- Upload PDF documents
- Automatically extract text
- Generate embeddings via Amazon Titan
- Store embeddings in Qdrant vector database
- View operation logs

### User Chat App
- Query uploaded documents
- Llama 3 via Bedrock for generating answers
- Context retention using LangChain memory
- Error handling and retry mechanisms
- Smooth UI/UX built with Streamlit

### Common Features
- Modular codebase
- Centralized logging
- Environment variable management
- Cloud service integrations (AWS, Qdrant)
- Automated tests with Pytest
- Ready for Docker and CI/CD pipelines

---

## 🏗️ Project Structure

```
ML-chatbot/
│
├── Admin/              # Admin portal for managing documents
│   ├── app.py
│   └── utils/          # Upload, embedding, vectorstore, logger
│
├── User/               # User-facing chat system
│   ├── main.py
│   ├── core/           # Chat logic, error handling, logging
│   └── services/       # LLM, memory, vector operations
│
│
├── .env                # AWS keys, Qdrant URL, Bedrock configs
├── .gitignore          # Ignore venv, cache, env, etc.
├── README.md           # Documentation
├── requirements.txt    # Required Python libraries
└── setup.py            # Installation and packaging file
```

---

## ⚙️ Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/2003saurabh/ML-chatbot.git
cd ML-chatbot
```

### 2. Create and Activate a Virtual Environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
.env\Scriptsctivate         # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup `.env` File
Create a `.env` file in the root folder:
```bash
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=your_aws_region
S3_BUCKET_NAME=your_bucket_name

QDRANT_URL=Qdrant_end_point
QDRANT_API_KEY=your_qdrant_key 

BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2
BEDROCK_LLM_MODEL_ID=meta.llama3-8b-instruct
```

---

## 🚀 How to Run the Applications

### Admin App (for uploading and managing documents)
```bash
cd Admin
streamlit run app.py
```

### User App (for chatting with documents)
```bash
cd User
streamlit run main.py
```

---


## 📚 Tech Stack

| Category | Technologies Used |
|:---------|:-------------------|
| UI | Streamlit |
| Backend | Python, LangChain |
| Cloud Services | AWS S3, Amazon Bedrock |
| Vector Store | Qdrant |
| Embeddings | Amazon Titan Text Embeddings V2 |
| LLM | Llama 3 (8B Instruct) |
| Memory Management | LangChain Memory |
| Testing | Pytest |
| Logging | Python logging module |
| Others | dotenv, requests, boto3 |

---

## 🛠️ Important Design Highlights

- **Modular Structure**: Clean separation between Admin, User, services, and core logic.
- **Environment Based**: No hard-coded credentials. Fully `.env` driven.
- **Resilient Uploads and Embeddings**: Retry and fallback mechanisms implemented.
- **Efficient Memory Handling**: LangChain memory allows coherent multi-turn conversations.
- **Testing Ready**: Unit and integration tests to ensure system reliability.

---

## 🔥 Future Enhancements

- 🔒 Secure User Authentication (SignUp/Login)
- 📈 Analytics Dashboard for document/chat insights
- 🐳 Dockerization for easy deployment
- 🌍 Multi-language support (RAG across languages)
- ⏳ Asynchronous Embedding Generation (for large docs)

---

## 🤝 Contribution Guidelines

1. Fork the repository 🍴
2. Create your branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Create a pull request 🔥

---

## 🙋‍♂️ Frequently Asked Questions (FAQ)

- **Q1: Can I add new LLMs easily?**
  - Yes! Just modify `llm_service.py` inside `User/services/`.

- **Q2: Where are embeddings stored?**
  - All document embeddings are pushed to Qdrant under a specified collection.

- **Q3: What happens if embedding generation fails?**
  - Admin utils handle retries and fallbacks automatically. Errors are logged in `admin.log`.

- **Q4: How is chat context preserved?**
  - User queries are linked via LangChain's memory modules.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

# ✨ Thank you for using **RAG-SELF** ✨
