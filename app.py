import os
import json
from datetime import datetime

import faiss
import numpy as np
import streamlit as st
import ollama
from sentence_transformers import SentenceTransformer


# ==========================================
# CONFIGURATION
# ==========================================

DOCUMENTS_DIR = "documents"
DATABASE_DIR = "database"
TICKETS_FILE = os.path.join(DATABASE_DIR, "tickets.json")

os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(DATABASE_DIR, exist_ok=True)


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Customer Support Agent",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Customer Support Agent")
st.caption("RAG-powered customer support with human escalation")


# ==========================================
# SESSION MEMORY
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# ==========================================
# LOAD DOCUMENTS
# ==========================================

@st.cache_data
def load_documents():

    documents = []

    if not os.path.exists(DOCUMENTS_DIR):
        return documents

    for filename in sorted(os.listdir(DOCUMENTS_DIR)):

        if filename.lower().endswith(".txt"):

            filepath = os.path.join(
                DOCUMENTS_DIR,
                filename
            )

            try:

                with open(
                    filepath,
                    "r",
                    encoding="utf-8"
                ) as file:

                    content = file.read().strip()

                if content:

                    documents.append(
                        {
                            "filename": filename,
                            "content": content
                        }
                    )

            except Exception:
                continue

    return documents


documents = load_documents()


# ==========================================
# CREATE FAISS VECTOR STORE
# ==========================================

@st.cache_resource
def create_vectorstore():

    docs = load_documents()

    if not docs:
        return None

    texts = [
        doc["content"]
        for doc in docs
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype(
        "float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    return index


index = create_vectorstore()


# ==========================================
# RETRIEVE RELEVANT DOCUMENTS
# ==========================================

def retrieve_documents(
    question,
    top_k=3
):

    if index is None or not documents:
        return []

    question_embedding = (
        embedding_model.encode(
            [question],
            convert_to_numpy=True
        )
        .astype("float32")
    )

    distances, indices = index.search(
        question_embedding,
        min(top_k, len(documents))
    )

    results = []

    for i in indices[0]:

        if 0 <= i < len(documents):

            results.append(
                documents[i]
            )

    return results


# ==========================================
# CREATE SUPPORT TICKET
# ==========================================

def create_ticket(question):

    if os.path.exists(TICKETS_FILE):

        try:

            with open(
                TICKETS_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                tickets = json.load(file)

        except (
            json.JSONDecodeError,
            FileNotFoundError
        ):

            tickets = []

    else:

        tickets = []

    ticket_id = (
        f"TICKET-{len(tickets) + 1:04d}"
    )

    ticket = {
        "ticket_id": ticket_id,
        "question": question,
        "status": "Open",
        "created_at": datetime.now().isoformat()
    }

    tickets.append(ticket)

    with open(
        TICKETS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            tickets,
            file,
            indent=4
        )

    return ticket_id


# ==========================================
# HUMAN ESCALATION DETECTION
# ==========================================

def user_requested_human(question):

    keywords = [
        "human",
        "agent",
        "customer care",
        "customer support",
        "support person",
        "talk to someone",
        "representative",
        "talk to human",
        "speak to human",
        "contact support"
    ]

    question = question.lower()

    return any(
        keyword in question
        for keyword in keywords
    )


# ==========================================
# DISPLAY CHAT HISTORY
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ==========================================
# CHAT INPUT
# ==========================================

question = st.chat_input(
    "Ask your customer support question..."
)


# ==========================================
# PROCESS USER QUESTION
# ==========================================

if question:

    # --------------------------------------
    # SAVE USER MESSAGE
    # --------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    # --------------------------------------
    # HUMAN SUPPORT REQUEST
    # --------------------------------------

    if user_requested_human(question):

        ticket_id = create_ticket(
            question
        )

        answer = (
            "👤 **Human support requested.**\n\n"
            "Your support ticket has been "
            "created successfully.\n\n"
            f"🎫 **Ticket ID:** `{ticket_id}`"
        )


    else:

        # ----------------------------------
        # RAG RETRIEVAL
        # ----------------------------------

        relevant_docs = retrieve_documents(
            question
        )


        # ----------------------------------
        # NO DOCUMENTS FOUND
        # ----------------------------------

        if not relevant_docs:

            ticket_id = create_ticket(
                question
            )

            answer = (
                "I couldn't find enough information "
                "in the company knowledge base.\n\n"
                "Your question has been escalated "
                "to human support.\n\n"
                f"🎫 **Ticket ID:** `{ticket_id}`"
            )


        else:

            # ----------------------------------
            # BUILD KNOWLEDGE CONTEXT
            # ----------------------------------

            context = "\n\n".join(
                [
                    (
                        f"Source: {doc['filename']}\n"
                        f"{doc['content']}"
                    )
                    for doc in relevant_docs
                ]
            )


            # ----------------------------------
            # CONVERSATION MEMORY
            # ----------------------------------

            history = ""

            for message in (
                st.session_state.messages[-6:]
            ):

                history += (
                    f"{message['role']}: "
                    f"{message['content']}\n"
                )


            # ----------------------------------
            # SYSTEM PROMPT
            # ----------------------------------

            system_prompt = """
You are an AI customer support agent.

Your job is to answer customer questions
using ONLY the provided company knowledge base.

IMPORTANT RULES:

1. Do not invent information.
2. Do not guess company policies.
3. Do not invent prices.
4. Do not invent refund rules.
5. Do not invent product information.
6. Do not use outside knowledge.
7. If the knowledge base does not contain
   enough information, respond exactly with:

ESCALATE_TO_HUMAN

8. Be helpful, concise and professional.
9. Answer directly using the available
   company information.
"""


            # ----------------------------------
            # USER PROMPT
            # ----------------------------------

            user_prompt = f"""
COMPANY KNOWLEDGE BASE:

{context}


CONVERSATION HISTORY:

{history}


CUSTOMER QUESTION:

{question}
"""


            # ----------------------------------
            # LOCAL LLM - OLLAMA
            # ----------------------------------

            try:

                response = ollama.chat(
                    model="llama3.2",
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ]
                )


                answer = response[
                    "message"
                ][
                    "content"
                ]


                # ----------------------------------
                # LOW CONFIDENCE ESCALATION
                # ----------------------------------

                if (
                    "ESCALATE_TO_HUMAN"
                    in answer
                ):

                    ticket_id = create_ticket(
                        question
                    )

                    answer = (
                        "I don't have enough "
                        "information to answer "
                        "that reliably.\n\n"
                        "I've escalated your request "
                        "to human support.\n\n"
                        f"🎫 **Ticket ID:** "
                        f"`{ticket_id}`"
                    )


            except Exception as e:

                answer = (
                    "⚠️ There was a problem "
                    "connecting to the local AI service.\n\n"
                    f"Error: {str(e)}"
                )


    # --------------------------------------
    # DISPLAY ASSISTANT ANSWER
    # --------------------------------------

    with st.chat_message(
        "assistant"
    ):

        st.markdown(answer)


    # --------------------------------------
    # SAVE ASSISTANT MESSAGE
    # --------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )