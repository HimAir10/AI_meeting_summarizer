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

## How to Run- 
- Just Run the app1.py, an html page will be triggered.
- On that page it will ask to enter video-Id, as i don't have any meeting transcript in my loacal machine, so i used the youtube transcript API.
- So, we will just need to Enter some Random youtube video ID, I tried with this "eHJnEHyyN1Y", you can also just copy this and paste it inside the tab, or if you want to continue with another video then follow this procedure -
- - Open youtube in your browser, open any podcast in english language and copy its address ID, just like this -
  - <img width="986" height="797" alt="Screenshot 2025-07-17 at 12 38 03 PM" src="https://github.com/user-attachments/assets/df5862b4-62f1-4c63-a2cd-213fb106c567" />
  - and paste this video Id on the tab and you are all set.
- And Inside the mail tab, I have only set the Gmail API permission for my personal mail, other people cannot use that, so if you want to access the mail feature to send someone in real then create your own Gmail-API request from Google from here -- https://console.cloud.google.com/
- - On this website follow these procedures to go ahead -
  - 1. Create or Select a Project
    -- Click Select Project > New Project
    -- Give it a name like YoutubeGmailBot

  - 2. Enable Gmail API
    -- Go to APIs & Services > Library
    -- Search for Gmail API
    -- Click Enable
    -- Give it a name like "YoutubeGmailBot"

  - 3. Configure OAuth Consent Screen
    -- Go to APIs & Services > OAuth consent screen
    -- Choose External
    -- Fill:
       -- App name: Youtube Chatbot
       -- User support email: your Gmail
       -- Developer contact: your Gmail
    -- Click Save and Continue until the end

  - 4. Add Yourself as a Test User
    -- In the same OAuth consent screen, under Test Users
    -- Click Add Users and enter your Gmail ("Your Email")

  - 5. Create OAuth 2.0 Client ID
    -- Go to APIs & Services > Credentials
    -- Click Create Credentials > OAuth client ID
    -- Application type: Desktop App
    -- name: anything (e.g. GmailLangchain)
    -- Click Create
    -- Click Download JSON
    -- Rename it to credentials.json
       
  - 6. Place credentials.json in the same folder as your Python script

## Demonstration - 

- The video Demonstration will give you the better Idea about the project.
  

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


