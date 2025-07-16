# Meeting Summarizer

This project is a full-stack application that takes meeting transcripts, generates summaries, highlights key objections and their resolutions, extracts action items, and optionally sends the results via email using Gmail API.

## 🚀 Features

- Extracts transcript from a video source (handled internally).
- Splits and embeds transcript text using advanced NLP techniques.
- Performs semantic retrieval using FAISS vector store.
- Uses a powerful LLM to:
  - Summarize meetings.
  - Extract objections and resolutions.
  - List action items and follow-ups.
- Presents results in a user-friendly HTML interface.
- Optionally sends the report to a user-provided email via Gmail integration.

---

## 🧠 Methodology

### 1. **Transcript Handling**
- Retrieves raw meeting transcript.
- Combines and prepares it for NLP processing.

### 2. **Text Preprocessing and Splitting**
- Uses `RecursiveCharacterTextSplitter` to divide the transcript into overlapping chunks to preserve context.

### 3. **Text Embedding**
- Applies `HuggingFaceEmbeddings` (using `thenlper/gte-small`) to transform text into dense vector embeddings.
- Embeddings are normalized for better cosine similarity performance.

### 4. **Vector Storage and Retrieval**
- Stores embeddings using `FAISS` with cosine similarity.
- Retrieves top-k relevant chunks for any question or prompt.

### 5. **Prompt-Based Querying**
- Uses `PromptTemplate` with different roles:
  - Summarization
  - Objection resolution
  - Action item extraction
- Executes `RetrievalQA` chains for each task using the LLM.

### 6. **LLM Integration**
- Uses `Google Gen-AI` with Langchain to call `gemini-2.0-flash`.
- Supports flexible prompting with temperature control.

### 7. **Email Sending (Optional)**
- Integrates with Gmail API via `langchain_google_community`.
- Allows sending the summarized report to any valid email.
- Requires `credentials.json` and `token.json` for OAuth2 authentication.

---

## 💡 Tech Stack

| Area             | Tool / Library                         |
|------------------|----------------------------------------|
| Backend          | Python (Flask)                         |
| Frontend         | HTML, CSS (Jinja Template)             |
| Vector DB        | FAISS                                  |
| LLM              | gemini-2.0-flash            |
| Embedding Model  | `thenlper/gte-small`                   |
| Email API        | Gmail API (`langchain_google_community`) |
| Text Processing  | LangChain, HuggingFace Transformers    |

---

## 🛠️ Installation

1. **Clone the repo:**
```bash
git clone https://github.com/HimAir10/AI_meeting_summarizer.git


