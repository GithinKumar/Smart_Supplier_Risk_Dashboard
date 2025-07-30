import json
import os
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.core import StorageContext, load_index_from_storage
import functools

def get_groq_client():
    import os
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set!")
    return Groq(api_key=api_key)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@functools.lru_cache(maxsize=1)
def get_client():
    return get_groq_client()

def load_dashboard_metadata():
    metadata_path = os.path.join(BASE_DIR, "Data", "dashboard_metadata.json")
    with open(metadata_path, 'r') as f:
        return json.load(f)
    
def load_supplier_delivery_metadata():
    metadata_path = os.path.join(BASE_DIR, "Data", "supplier_delivery_metadata.json")
    with open(metadata_path, 'r') as f:
        return json.load(f)
    
full_dashboard_metadata = load_dashboard_metadata()
full_supplier_delivery_metadata = load_supplier_delivery_metadata()

# ---- Load local vector index and embedding model ----
index_storage_path = os.path.join(BASE_DIR, "Data", "llama_index_storage")
if not os.path.exists(os.path.join(index_storage_path, "docstore.json")):
    raise RuntimeError("Vector index not found! Please run the vector index builder notebook or script first to create the index.")

@functools.lru_cache(maxsize=1)
def get_retriever():
    embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    storage_context = StorageContext.from_defaults(persist_dir=index_storage_path)
    index = load_index_from_storage(storage_context, embed_model=embed_model)
    return index.as_retriever()

def initialize_chat_history(session_state):
    if "chat_history" not in session_state:
        session_state["chat_history"] = []

def process_user_question(user_input, session_state):
    dashboard_metadata = get_relevant_metadata(user_input, full_dashboard_metadata, full_supplier_delivery_metadata)
    retriever = get_retriever()  # Now lazy-loads and caches!
    top_docs = retriever.retrieve(user_input)
    retrieved_chunks = "\n".join([doc.text for doc in top_docs])
    return ask_ai(user_input, session_state.get('chat_history', []), dashboard_metadata, context=retrieved_chunks)

def get_relevant_metadata(user_input, dashboard_metadata, delivery_metadata):
    import re
    user_input_lower = user_input.lower()
    summaries = {}

    # --- Flagged suppliers logic ---
    if "flagged" in user_input_lower or "flag" in user_input_lower:
        flagged = dashboard_metadata.get("Flagged_suppliers", {})
        summaries["flagged_suppliers_info"] = "\n".join(f"{k}: {v}" for k, v in flagged.items())
        
    # --- Lost shipments for a supplier logic ---
    match = re.search(r'sup\d+', user_input_lower)
    supplier_id = match.group(0).upper() if match else None
    if supplier_id and ("lost" in user_input_lower or "shipment" in user_input_lower):
        losts = [
            row for row in delivery_metadata.get("filtered_supplier_deliveries", [])
            if row.get("Supplier ID", "").upper() == supplier_id and row.get("Shipment Lost", False)
        ]
        if losts:
            lines = []
            for row in losts:
                lines.append(
                    f"- {row['Order Date']} (Expected: {row['Expected Delivery Date']}, Volume: {row['Shipment Volume']}, Value: {row['Value Category']})"
                )
            summaries["lost_shipments_info"] = (
                f"Supplier {supplier_id} lost shipments on:\n" + "\n".join(lines)
            )
    
    # Additional keyword filters for specific chart titles and supplier score
    if "supplier score" in user_input_lower:
        summaries["supplier_score"] = dashboard_metadata.get("ML Models Used-How score is Supplier Score is derived", "")

    if "average supplier score" in user_input_lower:
        summaries["chart_average_score"] = dashboard_metadata.get("Average Supplier Score (Bar Chart)", "")
    if "shipment volume" in user_input_lower or "volume per supplier" in user_input_lower:
        summaries["chart_shipment_volume"] = dashboard_metadata.get("Total Shipment Volume per Supplier (Bar Chart)", "")
    if "expected vs actual" in user_input_lower or "delivery line chart" in user_input_lower:
        summaries["chart_expected_vs_actual"] = dashboard_metadata.get("Expected vs Actual Deliveries (Multi-Series Line Chart with Event Markers)", "")
    if "supplier overview table" in user_input_lower:
        summaries["chart_supplier_overview"] = dashboard_metadata.get("Supplier Overview (Table)", "")
    
    # Fallback: include some generic context if no match
    if not summaries:
        chart_titles = [c['title'] for c in dashboard_metadata.get("charts", [])]
        summaries["dashboard_overview"] = "Available charts: " + ", ".join(chart_titles)
    return summaries

def ask_ai(prompt, history, dashboard_metadata=None, context="", model="llama-3.3-70b-versatile"):
    if dashboard_metadata:
        if isinstance(dashboard_metadata, dict):
            summaries_text = "\n".join(str(v) for v in dashboard_metadata.values())
        else:
            summaries_text = str(dashboard_metadata)
    else:
        summaries_text = ""
    
    system_prompt = (
    f"{summaries_text}\n"
    f"Relevant context from the dashboard:\n{context}\n"
    "You are Clippy, the intelligent assistant for the Supplier Performance dashboard — yes, that Clippy, the nostalgic paperclip from the Windows XP days. "
    "Now reborn with AI superpowers, you help users make sense of charts, filters, machine learning models, and supplier risk insights. "
    "You explain things clearly and concisely, like a friendly business analyst who enjoys adding a touch of humor and relatable examples.\n\n"
    "When asked about something complex, feel free to break it down with analogies — for example, compare the RAG pipeline to a smart handbook you flip through before answering a question.\n"
    "If you're asked something outside your scope, politely refuse and guide the user back to relevant dashboard topics.\n"
    "When referring to details not in the response, say: 'You can find more information in the GitHub repository.'\n"
    "Your tone should be informative, slightly witty, and always helpful. Avoid repeating context verbatim — interpret and explain like you're walking a colleague through it.\n"
    "If asked about supplier score weightage or scoring formula, only refer to the values explicitly defined in the metadata. Do not generalize or infer weightages based on external knowledge or assumptions.\n"
    )

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": prompt})

    client = get_client()  # Now lazy-loads and caches!
    chat_completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=520,
        temperature=0.6
    )
    response = chat_completion.choices[0].message.content

    # Post-process certain phrases in the assistant's output
    lower_resp = response.lower()
    if ("i don’t have that information" in lower_resp) or ("i don't have exact details" in lower_resp):
        response = "I'm not seeing this detail directly in the dashboard, but a more complete explanation is available in the GitHub repository."

    return response

