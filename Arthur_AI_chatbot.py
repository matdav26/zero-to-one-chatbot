import uuid
import requests
import streamlit as st
import openai
from pinecone import Pinecone

# Set API keys
openai.api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
arthur_token = st.secrets["ARTHUR_API_KEY"]
index_name = "podcastrag2"

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)

# Arthur URLs
ARTHUR_PROMPT_URL = "http://localhost:3030/api/v2/tasks/022bd0ba-366f-41f4-9bb0-cac09910b650/validate_prompt"
ARTHUR_RESPONSE_URL_TEMPLATE = "http://localhost:3030/api/v2/tasks/022bd0ba-366f-41f4-9bb0-cac09910b650/validate_response/{inference_id}"

HEADERS = {
    "Authorization": f"Bearer {arthur_token}",
    "Content-Type": "application/json"
}

def send_trace_to_arthur(prompt, response, context):
    try:
        payload_prompt = {
            "prompt": prompt,
            "user_id": "test-user",
            "conversation_id": str(uuid.uuid4())
        }
        prompt_res = requests.post(ARTHUR_PROMPT_URL, json=payload_prompt, headers=HEADERS)
        prompt_res.raise_for_status()
        inference_id = prompt_res.json()["inference_id"]

        payload_response = {
            "response": response,
            "context": context
        }
        response_url = ARTHUR_RESPONSE_URL_TEMPLATE.format(inference_id=inference_id)
        response_res = requests.post(response_url, json=payload_response, headers=HEADERS)
        response_res.raise_for_status()

    except Exception as e:
        print("⚠️ Arthur error:", e)

def answer_query(query, index, top_k=5, model="gpt-4o"):
    # Embed query
    embedding_response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = embedding_response.data[0].embedding

    # Pinecone search
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    # Context for prompt
    context = "\n\n".join([
        f"{m['metadata'].get('speaker', 'Unknown')}: {m['metadata']['text']}"
        for m in results["matches"]
    ])

    # Individual excerpts for UI display
    sources = "\n\n".join([
        f"- **{m['metadata'].get('speaker', 'Unknown')}**: {m['metadata']['text']}"
        for m in results["matches"]
    ])

    prompt = f"""
    You are a helpful assistant trained on podcast transcripts from the podcast 'Zero to One'. 
    Each excerpt includes content spoken by a specific guest or co-host. 
    When answering the question, always include the name of the speaker. 
    Do not refer to "a guest" or "another speaker" — instead, say "Jonathan said..." or "Ilan mentioned that..." etc.
    When answering, never say 'a guest' or 'another speaker'. Always use the speaker's name exactly as shown in the excerpts.

    Provide a summary of each answer you are providing with bullet points.

    Use the following excerpts to answer the user's question.

    Excerpts:
    {context}

    Question: {query}
    Answer:
    """

    chat_response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    answer = chat_response.choices[0].message.content.strip()

    send_trace_to_arthur(query, answer, context)

    return answer, context
