import uuid
import requests
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone

# Retrieve secrets using Streamlit's secure storage
openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
arthur_token = st.secrets["ARTHUR_API_KEY"]
index_name = "podcastrag2"

# Initialize OpenAI and Pinecone clients
client = OpenAI(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)

# Arthur validation URLs
ARTHUR_PROMPT_URL = "http://localhost:3030/api/v2/tasks/022bd0ba-366f-41f4-9bb0-cac09910b650/validate_prompt"
ARTHUR_RESPONSE_URL_TEMPLATE = "http://localhost:3030/api/v2/tasks/022bd0ba-366f-41f4-9bb0-cac09910b650/validate_response/{inference_id}"

HEADERS = {
    "Authorization": f"Bearer {arthur_token}",
    "Content-Type": "application/json"
}

def send_trace_to_arthur(prompt, response, context):
    payload_prompt = {
        "prompt": prompt,
        "user_id": "test-user",
        "conversation_id": str(uuid.uuid4())
    }

    try:
        prompt_res = requests.post(ARTHUR_PROMPT_URL, json=payload_prompt, headers=HEADERS)
        prompt_res.raise_for_status()
        prompt_data = prompt_res.json()
        inference_id = prompt_data["inference_id"]
        print("✅ Arthur prompt validated:", prompt_data)

        payload_response = {
            "response": response,
            "context": context
        }
        response_url = ARTHUR_RESPONSE_URL_TEMPLATE.format(inference_id=inference_id)
        response_res = requests.post(response_url, json=payload_response, headers=HEADERS)
        response_res.raise_for_status()
        print("✅ Arthur response validated:", response_res.json())

    except Exception as e:
        print("⚠️ Arthur error:", e)

def answer_query(query, index, top_k=5, model="gpt-4o"):
    # Step 1: Embed the query
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    query_embedding = response.data[0].embedding

    # Step 2: Search Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    # Step 3: Build the context
    context = "\n\n".join([
        f"{m['metadata'].get('speaker', 'Unknown')}: {m['metadata']['text']}"
        for m in results["matches"]
    ])

    # Step 4: Build the prompt
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

    # Step 5: Call GPT
    chat_response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    answer = chat_response.choices[0].message.content.strip()

    # Step 6: Send to Arthur
    send_trace_to_arthur(query, answer, context)

    return answer
