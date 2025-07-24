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
        flagged_suppliers = flagged.get("suppliers flagged", "N/A")
        sup003_quarters = flagged.get("SUP003_Flagged_quarters", "N/A")
        sup005_quarters = flagged.get("SUP005_Flagged_quarters", "N/A")
        summaries["flagged_suppliers_info"] = (
            f"The flagged suppliers in this dashboard are SUP003 (flagged in {sup003_quarters}) "
            f"and SUP005 (flagged in {sup005_quarters})."
            )
        
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
    
    # Fallback: include some generic context if no match
    if not summaries:
        chart_titles = [c['title'] for c in dashboard_metadata.get("charts", [])]
        summaries["dashboard_overview"] = "Available charts: " + ", ".join(chart_titles)
    return summaries

full_metadata = load_dashboard_metadata()

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
        "You are an assistant for the Supplier Performance dashboard. "
        "Only discuss the charts, filters, data, and ML models in this dashboard. "
        "If asked about anything else, politely refuse."
    )

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": prompt})

    client = get_client()  # Now lazy-loads and caches!
    chat_completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=200,
        temperature=0.6
    )
    return chat_completion.choices[0].message.content
