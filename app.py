import os
import shutil
from flask import Flask, render_template, request, session, redirect, url_for
from flask_session import Session
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Flask setup
app = Flask(__name__)
app.secret_key = "your_secret_key"

# Server-side session config
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = "./flask_session"
app.config["SESSION_PERMANENT"] = False
Session(app)

# Clear old session files on server restart
if os.path.exists("./flask_session"):
    shutil.rmtree("./flask_session")
os.makedirs("./flask_session", exist_ok=True)

# Retriever cache (so we don't reprocess transcript each time)
retriever_cache = {}

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html",
        history=session.get("chat_history", []),
        session_video_id=session.get("video_id")
    )

@app.route("/predict", methods=["POST"])
def predict():
    question = request.form.get("question", "").strip()
    incoming_video_id = request.form.get("video_id")

    # Set new video ID if given and reset chat
    # Only reset chat if video ID has changed
    if incoming_video_id and incoming_video_id != session.get("video_id"):
        session["video_id"] = incoming_video_id
        session["chat_history"] = []

    video_id = session.get("video_id")
    if not video_id:
        return render_template("index.html", answer="Please provide a video ID.", history=[], session_video_id=None)



    if video_id not in retriever_cache:
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
            transcript = " ".join(chunk["text"] for chunk in transcript_list)
        except TranscriptsDisabled:
            return render_template("index.html", answer="Transcripts are disabled.", history=[], session_video_id=None)
        except Exception as e:
            return render_template("index.html", answer=f"Error: {str(e)}", history=[], session_video_id=None)

        # Text splitting
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])

        # Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="thenlper/gte-small",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        # Build retriever
        vector_store = FAISS.from_documents(chunks, embeddings, distance_strategy=DistanceStrategy.COSINE)
        retriever_cache[video_id] = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    retriever = retriever_cache[video_id]

    # Prompt template
    prompt1 = PromptTemplate(
        template="""You are a helpful assistant. Summarize the content of the video in well defined words and easy explanations.
If the context is insufficient, say you can't summarize.

{context}

Question: {question}""",
        input_variables=["context", "question"]
    )

    # LLM model
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )

    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    final_prompt = prompt1.invoke({"context": context, "question": question})
    response = model.invoke(final_prompt)

    # Update chat history
    if "chat_history" not in session:
        session["chat_history"] = []

    session["chat_history"].append({
        "user": question,
        "bot": response.content
    })

    return render_template("index.html",
        history=session["chat_history"],
        session_video_id=video_id
    )

@app.route("/reset", methods=["POST"])
def reset():
    session.clear()
    global retriever_cache
    retriever_cache = {}
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)
