# 📊 Supplier Risk Dashboard

An intelligent, AI-powered dashboard for assessing and visualizing supplier risk.

By leveraging machine learning on financial and delivery data, the platform generates a supplier risk score that provides both high-level overviews and in-depth analytical insights through interactive visualizations. Additionally, a built-in AI assistant enables users to ask questions, explore data, and receive guided support throughout their analysis.

---

## 🧠 Key Features

- 📊 **Quarterly Supplier Scoring**  
  ML-generated single-value matrix to compare suppliers against each other.

- 📉 **Interactive Dashboard**
  - Value Category Filtering (Critical, High, Medium, Low, All)
  - Supplier Filtering
  - Timeline Filtering
  - Volume-based Bar Charts
  - Expected vs Actual Delivery Line Chart

- 📋 **Supplier Overview Table**  
  Quick glance at supplier delivery performance.

---

## 📂 Project Structure

```
supplier-risk-dashboard/
├── data/                  # Raw and processed datasets
├── notebooks/             # Jupyter notebooks for scoring/testing
├── src/                   # Model training, data processing, and dashboard scripts
├── requirements.txt       # Project dependencies
├── README.md              # You're reading it!
└── .vscode/               # VS Code config
```
</code></pre>


## 📈 Dashboard - ML & LLM Workflow
<img width="417" height="603" alt="image" src="https://github.com/user-attachments/assets/f3aa7f57-9564-4c7b-bb4c-576e429b1745" />


### 1. Dataset Ingestion and Structure
Curated datasets are used for ML models and RAG pipelines:

- **Supplier Master Dataset:**  
  `Supplier ID`, `Name`, `Tier`, `Region`
- **Supplier Financials Dataset:**  
  `Supplier ID`, `Quarter`, `Credit Score`, `Revenue`, `D&B Rating`
- **Supplier Delivery Dataset:**  
  `Supplier ID`, `Order Date`, `Expected vs Actual Delivery`, `Shipment Lost`, `Defect Rate`, `Volume`, `Value Category`

---

### 2. Data Aggregation and Enrichment

- **Supplier Overview Dataset:**  
  Merge of `supplier_master_dataset` and `supplier_delivery_dataset` for a holistic delivery performance profile.
- **Delivery Metadata:**  
  Derived from the overview dataset for RAG workflows.

---

### 3. Unsupervised Anomaly Detection (Isolation Forest)

- **Input:** `supplier_financials_dataset`  
- **Model:** Isolation Forest  
- **Output:** `supplier_financial_risk_score`
  - Flags anomalies in credit, revenue, or D&B rating.
  - Acts as a soft signal.

---

### 4. Supervised Risk Scoring (XGBoost Regressor)

- **Inputs:**
  - `supplier_overview`
  - `supplier_master_dataset`
  - `supplier_financial_risk_score`- as a minor signal
  - `supplier_finacials_dataset`
- **Model:** XGBoost Regressor
- **Scoring Formula:**

```python
    y = (
        0.2256 * (100 - merged_df["%_shipment_lost"]) +
        0.188 * (100 - merged_df["%_defect_rate"]) +
        0.1504 * (100 - merged_df["%_delayed"]) +
        0.1353 * (100 - merged_df["avg_delay"]) +
        0.0375 * (100 - merged_df["Financial Risk Score"]) +
        0.1128 * (merged_df["Credit Score"]) +
        0.0752 * (100 - merged_df["D&B Rating"]) +
        0.0752 * (100 - merged_df["Log Revenue"])
    )
    
```
- **Business Logic:**
  - Emphasis on shipment lost and defect rate
  - Financial risk score used as a minor signal

---

### 5. ML Metadata Embedding for RAG

- Model outputs and metadata are embedded for contextual use  
- Used for vectorization and AI augmentation in RAG pipelines

---

### 6. RAG Pipeline for AI Contextualization

- **Embedding Model:** `all-MiniLM-L6-v2`  
- **LLM Application:** LLaMA 3.3 70B-powered chatbot (**Clippy**)  
- **Capabilities:**
  - Context-aware Q&A  
  - Scoring logic explanations  
  - Intelligent data exploration  

---

### 7. Streamlit-Based Enterprise Dashboard

- ML-driven Supplier Risk Scorecards  
- Standard KPI Visualizations  
- **Integrated AI Chatbot:**
  - Supplier Q&A  
  - Scoring rationale  
  - Document-based Q&A  

---

## 🏢 Integrating ML and LLMs in an Organizational Workflow

1. **Raw Data Ingestion**  
   Sources: Supply chain logs, financials, ERP/CRM exports, etc.

2. **Data Lake Storage**  
   Platforms: Amazon S3, Azure Blob, GCS

3. **ETL & Orchestration**  
   - Tools: Airflow, Prefect  
   - Tasks: Cleaning, validation, schema unification, scheduling

4. **Analytical & Hybrid Storage**
   - Structured: PostgreSQL, Snowflake, BigQuery  
   - Semi-structured: MongoDB, DynamoDB

5. **ML & LLM Layer**
   - ML: Forecasting, scoring, anomaly detection  
   - LLM: RAG workflows using vector stores (FAISS, Pinecone)  
   - Feature Store + Fine-Tuning  

6. **Enterprise UI Layer**
   - Risk scoring & KPI dashboards  
   - LLM-powered AI modules  
   - Context-aware chatbot  
   - What-If simulations  
   - Document intelligence  
   - Natural language search  

---

## 🧪 LLM Strategy

To support intelligent risk recommendations:

- ⚖️ **Embeddings:** `all-MiniLM-L6-v2` or OpenAI vectors  
- 📊 **RAG Queries:** Over financials, documents, and emails  
- 🧩 **Stage 1:** Prompt engineering + template grounding  
- 🧠 **Stage 2:** LLM fine-tuning for domain-specific Q&A  
