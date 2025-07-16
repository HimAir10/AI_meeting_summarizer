import os
from flask import Flask, render_template, request, session
from flask_session import Session
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_google_community import GmailToolkit
from langchain.agents import create_react_agent, AgentExecutor


from langchain_google_community.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,
)

# Can review scopes here https://developers.google.com/gmail/api/auth/scopes
# For instance, readonly scope is 'https://www.googleapis.com/auth/gmail.readonly'
credentials = get_gmail_credentials(
    token_file="token.json",
    scopes=["https://mail.google.com/"],
    client_secrets_file="credentials.json",
)
api_resource = build_resource_service(credentials=credentials)
toolkit = GmailToolkit(api_resource=api_resource)

tools = toolkit.get_tools()
to_ol = tools[1]
# Load environment variables
load_dotenv()

# Flask setup
app = Flask(__name__)
app.secret_key = "your_secret_key"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Cache to store retrievers per video ID
retriever_cache = {}

@app.route("/", methods=["GET"])
def home():
    return render_template("index1.html", results=None, session_video_id=session.get("video_id"))

@app.route("/predict", methods=["POST"])
def predict():
    incoming_video_id = request.form.get("video_id", "").strip()

    if not incoming_video_id:
        return render_template("index1.html", error="Please provide a video ID.")

    session["video_id"] = incoming_video_id
    video_id = session["video_id"]

    if video_id not in retriever_cache:
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
            transcript = " ".join(chunk["text"] for chunk in transcript_list)
        except TranscriptsDisabled:
            return render_template("index1.html", error="Transcripts are disabled for this video.")
        except Exception as e:
            return render_template("index1.html", error=f"Error fetching transcript: {str(e)}")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])

        embeddings = HuggingFaceEmbeddings(
            model_name="thenlper/gte-small",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings,
            distance_strategy=DistanceStrategy.COSINE
        )
        retriever_cache[video_id] = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

    retriever = retriever_cache[video_id]

    from langchain_google_genai import ChatGoogleGenerativeAI

    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )

    # Prompt templates
    prompts = {
        "summary": """You are a helpful assistant. Summarize the content of the video in clear and concise language.\n\n{context}""",
        "objections": """You are a helpful assistant. Highlight any objections and how they were resolved in the content.\n\n{context}""",
        "actions": """You are a helpful assistant. Extract action items or follow-ups discussed in the video.\n\n{context}"""
    }

    results = {}
    for key, prompt_text in prompts.items():
        qa_chain = RetrievalQA.from_chain_type(
            llm=model,
            retriever=retriever,
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template=prompt_text,
                    input_variables=["context"]
                )
            }
        )
        output = qa_chain.run("")
        results[key] = output

    session['results'] = results
    return render_template("index1.html", results=results, session_video_id=video_id)

@app.route("/send_email", methods=["POST"])
def send_email():
    recipient_email = request.form.get("email_to", "").strip()
    if not recipient_email:
        return render_template("index1.html", error="Recipient email is required.", results=session.get("results"))

    results = session.get("results")
    if not results:
        return render_template("index1.html", error="No report found to send.")

    # Compose email body
    message_body = (
        f"🔍 Summary:\n{results['summary']}\n\n"
        f"⚠️ Objections:\n{results['objections']}\n\n"
        f"✅ Action Items:\n{results['actions']}\n"
    )

    try:
        to_ol.invoke({
            "to": f'{recipient_email}',
            "subject": "YouTube Meeting Report",
            "message": f'{message_body}'
        })
        confirmation = f"Email sent successfully to {recipient_email}"
        return render_template("index1.html", results=results, confirmation=confirmation)
    except Exception as e:
        return render_template("index1.html", results=results, error=f"Email failed: {str(e)}")



if __name__ == "__main__":
    app.run(debug=True)
