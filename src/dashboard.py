import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
import os
from AI_Chatbot import initialize_chat_history, process_user_question
from streamlit_lottie import st_lottie
import json
import time

# ----------------------------- ONBOARDING LOGIC  -----------------------------
 
if "show_onboarding" not in st.session_state:
    st.session_state["show_onboarding"] = True

st.set_page_config(layout="wide")

# Helper function to load Lottie JSON
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Load Lottie avatars at the top (once)
Clippy_blush = load_lottiefile("src/lottie_avatars/Clippy/Clippy_Blush_0_75.json")
Clippy_pointing = load_lottiefile("src/lottie_avatars/Clippy/Clippy_Pointing_75_140.json")
Clippy_pointing2 = load_lottiefile("src/lottie_avatars/Clippy/Clippy_Pointing_.3.435_552.json")
Clippy_ideal_onboarding = load_lottiefile("src/lottie_avatars/Clippy/Clippy Animation.json")
Clippy_ideal_sidebar = load_lottiefile("src/lottie_avatars/Clippy/Clippy_ideal_sidebar.json")
Clippy_checkmark = load_lottiefile("src/lottie_avatars/Clippy/Clippy_Checkmark_177_435.json")
Clippy_researching = load_lottiefile("src/lottie_avatars/Clippy/Clippy_Research_552_1057.json")
Loader = load_lottiefile("src/lottie_avatars/Lamma.json")
ML = load_lottiefile("src/lottie_avatars/ML.json")
AI = load_lottiefile("src/lottie_avatars/AI pipeline.json")

# Onboarding screen
# slide tracking
if "onboarding_slide" not in st.session_state:
    st.session_state["onboarding_slide"] = 1

if st.session_state["show_onboarding"]:
    slide = st.session_state["onboarding_slide"]

    if slide == 1:
        col_left, col_right = st.columns([1.15, 2.2])
        with col_left:
            st.markdown("<div style='height:180px;'></div>", unsafe_allow_html=True)
            st.markdown("<div style='margin-left:-90px;'>", unsafe_allow_html=True)
            st_lottie(Clippy_ideal_onboarding, height=250, key="onboarding_clippy")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_right:
            st.markdown("<div style='height:70px;'></div>", unsafe_allow_html=True)
            st.markdown("""
                <div style='
                    background: white; color: #222; padding: 32px 36px;
                    border-radius: 18px; max-width: 600px; text-align: justify; 
                    box-shadow: 0 8px 32px #0003; margin-top: 24px;
                    font-size: 18px;
                '>
                <h3 style='text-align:center; margin-bottom: 18p;'>
                Hi, I am Clippy, your AI assistant for this Dashboard!</h3>
                <p>
                I am here to walk you through 
                <span style="font-size:1.11em; font-weight:800; color:#101e4c;">
                Githin Kumar's Supplier Risk Assessment Dashboard</span>.<br>
                <br>
                This project combines the power of advanced analytics, interactive BI dashboards, 
                and cutting-edge AI (ML & LLM/SLM) to deliver a new level of supplier risk insight. 
                Unlike traditional BI solutions, this approach goes beyond static reports harnessing 
                machine learning and language models and can be designed to detect complex patterns, 
                provide tailored recommendations, and adapt to data's unique context.
                <br><br>     
                Please click on <b>Next</b> to continue.
                </p>
                </div>
            """, unsafe_allow_html=True)
            cols = st.columns([1.15,0.75])
            with cols[0]:
                st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True) 
                if st.button("Skip", key="onboarding_skip_1"):
                    st.session_state["show_onboarding"] = False
                    st.rerun()
                
            with cols[1]:
                st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True) 
                if st.button("Next", key="onboarding_next_1"):
                    st.session_state["onboarding_slide"] = 2
                    st.rerun()
        st.stop()
                
            
    elif slide == 2:
        col_left, col_right = st.columns([1.15, 2.2])
        with col_left:
            st.markdown("<div style='height:180px;'></div>", unsafe_allow_html=True)
            st.markdown("<div style='margin-left:-90px;'>", unsafe_allow_html=True)
            st.lottie(ML, height=300, key="coding_anim")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_right:
            st.markdown("<div style='height:70px;'></div>", unsafe_allow_html=True)
            st.markdown("""
                <div style='
                    background: white; color: #222; padding: 32px 36px;
                    border-radius: 18px; max-width: 600px; text-align: justify; 
                    box-shadow: 0 8px 32px #0003; margin-top: 24px;
                    font-size: 18px;
                '>
                This dashboard delivers a comprehensive view of supplier delivery records, shipment volumes, supplier scores, 
                risk flagging, and flexible filters for easy comparison and performance assessment.
                </p>
                Supplier scores are powered by ML models such as Isolation Forest and XGBRegressor, which analyze 
                delivery, financial, and supplier overview data to identify and flag potential risks.
                </p>    
                Real-world solutions can integrate data sources like email archives, contract documents, 
                invoices, purchase orders, performance reviews, and support tickets. By leveraging 
                deep learning transformer models and LLMs, to enable advanced pattern detection, 
                predictive analytics, automated document analysis, and enchanced supplier risk management.
                </p>
                </div>
            """, unsafe_allow_html=True)
            cols = st.columns([1.15,.8])
            with cols[0]:
                st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True) 
                if st.button("Skip", key="onboarding_skip_2"):
                    st.session_state["show_onboarding"] = False
                    st.rerun()
                
            with cols[1]:
                st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True) 
                if st.button("Next", key="onboarding_next_2"):
                    st.session_state["onboarding_slide"] = 3
                    st.rerun()
        st.stop()

    elif slide == 3:
        col_left, col_right = st.columns([1.15, 2.2])
        with col_left:
            st.markdown("<div style='height:180px;'></div>", unsafe_allow_html=True)
            st.markdown("<div style='margin-left:-90px;'>", unsafe_allow_html=True)
            st.lottie(AI, height=300, key="pipeline_anim")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_right:
            st.markdown("<div style='height:70px;'></div>", unsafe_allow_html=True)
            st.markdown("""
                <div style='
                    background: white; color: #222; padding: 32px 36px;
                    border-radius: 18px; max-width: 600px; text-align: justify; 
                    box-shadow: 0 8px 32px #0003; margin-top: 24px;
                    font-size: 18px;
                '>
                This dashboard features an integrated AI assistant, Clippy, trained on project-specific data. 
                Clippy eliminates the need for a dedicated presenter or dashboard owner by instantly answering questions, 
                explaining dashboard elements, and guiding users making onboarding easier for new staff and providing valuable support 
                for management or cross-departmental reviews.
                </p>
                This particular chatbot is powered by llama 3.3 70b versatile model from Groq with RAG pipeline for context awarness.
                The model is highly capable, but it is running on a free tier of Groq's API which comes with its limitations.
                </p>    
                In real-world scenarios, companies can deploy advanced, locally-run models such as 
                Moonshot AI Kimi K2, Llama 3, Mistral, or Vicuna to ensure privacy and customization. 
                There is also a GitHub repository with a complete codebase for this project.
                </p>
                </div>
        """, unsafe_allow_html=True)
        cols = st.columns([1.15,.8])
        with cols[1]:
            st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
            if st.button("Finish", key="onboarding_finish"):
                st.session_state["show_onboarding"] = False
                st.rerun()

        st.stop()

# ----------------------------- LOAD DATA AND INIT -----------------------------
# Load all data files at the top, for global use
delivery_df = pd.read_csv("Data/3.supplier_delivery_dataset.csv")
delivery_df["Value Category"] = delivery_df["Value Category"].fillna("Unknown")

master_df = pd.read_csv("Data/1.supplier_master_dataset.csv")
supplier_name_map = dict(zip(master_df["Supplier ID"], master_df["Supplier Name"]))

summary = pd.read_csv("Data/7.supplier_delivery_table.csv")

df = pd.read_csv("Data/6.supplier_score_final.csv")
df["Quarter"] = pd.to_datetime(df["Quarter"])

overview_df = pd.read_csv("Data/7.supplier_delivery_table.csv")


flagged_score_threshold = 70
# Filter rows where score is low
flagged_scores_df = df[df['Supplier Score'] < flagged_score_threshold]
# Group flagged quarters by supplier
flagged_quarters = (
    flagged_scores_df.groupby('Supplier ID')['Quarter']
    .apply(lambda qs: ', '.join(f"{q.year}-Q{((q.month-1)//3)+1}" for q in qs))
    .reset_index()
    .rename(columns={'Quarter': 'Flagged Quarters'})
)
flagged_suppliers = (
    master_df[master_df['Supplier ID'].isin(flagged_scores_df['Supplier ID'].unique())]
    [['Supplier ID', 'Supplier Name']]
    .merge(flagged_quarters, on='Supplier ID', how='left')
)

shipment_categories = ["All", "Critical", "High", "Low", "Medium"]
supplier_ids = sorted(df["Supplier ID"].unique())
supplier_options = ["ALL"] + [f"{sid} - {supplier_name_map.get(sid, '')}" for sid in supplier_ids]

flagged_supplier_id = "SUP005"
flagged_supplier_name = supplier_name_map.get(flagged_supplier_id, "")
default_supplier_display = f"{flagged_supplier_id} - {flagged_supplier_name}"
default_supplier_idx = supplier_options.index(default_supplier_display) if default_supplier_display in supplier_options else 0

# Initialize chat history
initialize_chat_history(st.session_state)

#  filters block to and Clippy AI
st.markdown("<h1 style='text-align: center;'>SMART SUPPLIER OVERVIEW</h1>", unsafe_allow_html=True)

if not st.session_state["show_onboarding"]:
    with st.sidebar:
        with st.expander("Filters", expanded=True):
            colA, colB = st.columns(2)
            with colA:
                shipment_category = st.selectbox("Select Value Category", shipment_categories, index=0)
            with colB:
                supplier_selected_display = st.selectbox("Select Supplier", supplier_options, index=default_supplier_idx)
            supplier_selected = supplier_selected_display.split(" - ")[0] if supplier_selected_display != "ALL" else "ALL"

            filtered_df = delivery_df.copy()
            filtered_df['Expected Delivery Date'] = pd.to_datetime(filtered_df['Expected Delivery Date'])
            filtered_df['Actual Delivery Date'] = pd.to_datetime(filtered_df['Actual Delivery Date'])

            if supplier_selected != "ALL":
                filtered_df = filtered_df[filtered_df['Supplier ID'] == supplier_selected]
            if shipment_category != "All":
                filtered_df = filtered_df[filtered_df['Value Category'] == shipment_category]

            min_date = filtered_df['Expected Delivery Date'].min().to_pydatetime().date()
            max_date = filtered_df['Expected Delivery Date'].max().to_pydatetime().date()
            date_range = st.slider(
                "Select Date Range",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date),
                format="YYYY-MM-DD"
            )
            filtered_df = filtered_df[
                (filtered_df['Expected Delivery Date'].dt.date >= date_range[0]) &
                (filtered_df['Expected Delivery Date'].dt.date <= date_range[1])
            ]


# ----------------------------- CARDS: SUPPLIER SUMMARY BLOCK -----------------------------
st.write("## Supplier Summary")
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Suppliers", master_df['Supplier ID'].nunique())
with col2:
    st.metric("Flagged Suppliers", flagged_suppliers['Supplier ID'].nunique())

# Show flagged suppliers below cards
if not flagged_suppliers.empty:
    st.write("### Flagged Suppliers")
    st.dataframe(flagged_suppliers.reset_index(drop=True), hide_index=True)
else:
    st.write("No flagged suppliers detected.")


# ----------------------------- SIDEBAR: AI CHATBOT BLOCK -----------------------------

if not st.session_state["show_onboarding"]:

    with st.sidebar:

        # Initialize chatbot state variables
        if "chatbot_state" not in st.session_state:
            st.session_state["chatbot_state"] = "ideal"
        if "last_interaction_time" not in st.session_state:
            st.session_state["last_interaction_time"] = 0
        if "processing_message" not in st.session_state:
            st.session_state["processing_message"] = False

        state = st.session_state["chatbot_state"]

        # Display Clippy reaction according to state
        if state == "blush":
            st_lottie(Clippy_blush, height=125, key="clippy_blush")
        elif state == "pointing":
            st_lottie(Clippy_pointing, height=125, key="clippy_pointing")
        elif state == "pointing2":
            st_lottie(Clippy_pointing2, height=125, key="clippy_pointing2")
        elif state == "checkmark":
            st_lottie(Clippy_checkmark, height=125, key="clippy_checkmark")
        elif state == "researching" and st.session_state.get("processing_message", False):
            st_lottie(Clippy_researching, height=125, key="clippy_researching")
        else:  # fallback/default
            st_lottie(Clippy_ideal_sidebar, height=125, key="clippy_ideal_sidebar")

        st.header("Clippy AI")

        # --- Chatbot chat history window ---
        chat_history_html = ""
        for msg in st.session_state.get("chat_history", []):
            if msg["role"] == "user":
                chat_history_html += f'<div style="color:#fff;background:#444;padding:6px 12px;margin-bottom:5px;border-radius:8px;text-align:right;">You: {msg["content"]}</div>'
            else:
                chat_history_html += f'<div style="color:#222;background:#dedede;padding:6px 12px;margin-bottom:5px;border-radius:8px;text-align:left;">Clippy: {msg["content"]}</div>'

        st.markdown(f"""
        <div style="
            max-height: 700px;
            overflow-y: auto;
            border: 1px solid #444;
            background: #222;
            padding: 8px;
            border-radius: 10px;
            margin-bottom: 10px;
            height: 450px;
        ">
            {chat_history_html}
        </div>
        """, unsafe_allow_html=True)
        
        # Show loader only when actually processing
        if st.session_state.get("processing_message", False):
            st_lottie(Loader, height=70, key="ai_loader")
            st.write("Processing your question...")

        user_input = st.chat_input("Ask me about the dashboard…")

        # Handle user input ONLY when submitted
        if user_input and not st.session_state.get("processing_message", False):
            # Set processing state
            st.session_state["processing_message"] = True
            st.session_state["chatbot_state"] = "researching"
            
            # Add user message to chat history
            st.session_state["chat_history"].append({"role": "user", "content": user_input})
            
            # Rerun to show processing state
            st.rerun()

        # Process the message if we're in processing state
        if st.session_state.get("processing_message", False) and len(st.session_state["chat_history"]) > 0:
            if st.session_state["chat_history"][-1]["role"] == "user":
                last_user_message = st.session_state["chat_history"][-1]["content"]
                
                # Process the AI response
                try:
                    ai_response = process_user_question(last_user_message, st.session_state)
                    
                    # Add AI response to chat history
                    st.session_state["chat_history"].append({"role": "assistant", "content": ai_response})
                    
                    # Reset processing state
                    st.session_state["processing_message"] = False
                    st.session_state["chatbot_state"] = "checkmark"
                    
                    # Auto-revert to ideal state after showing checkmark
                    time.sleep(1)  # Brief pause to show checkmark
                    st.session_state["chatbot_state"] = "ideal"
                    
                    st.rerun()
                    
                except Exception as e:
                    st.session_state["chat_history"].append({"role": "assistant", "content": f"Sorry, I encountered an error: {str(e)}"})
                    st.session_state["processing_message"] = False
                    st.session_state["chatbot_state"] = "ideal"
                    st.rerun()

# ----------------------------- AVERAGE SUPPLIER SCORE BAR CHART BLOCK -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.write("## Average Supplier Score")
    quarters_in_range = df[
        (df["Quarter"].dt.date >= date_range[0]) &
        (df["Quarter"].dt.date <= date_range[1])
        ]
    avg_scores = quarters_in_range.groupby("Supplier ID")["Supplier Score"].mean().reset_index()
    fig_bar = px.bar(
        avg_scores,
        x="Supplier ID",
        y="Supplier Score",
        color="Supplier Score",
        color_continuous_scale="RdYlGn",
        title="Average Supplier Score"
        )
    st.plotly_chart(fig_bar, use_container_width=True)

# ----------------------------- TOTAL SHIPMENT VOLUME BAR CHART BLOCK -----------------------------
with col2:
    st.write("## Total Shipment Volume per Supplier")

    shipment_volume_df = delivery_df.copy()
    shipment_volume_df['Expected Delivery Date'] = pd.to_datetime(shipment_volume_df['Expected Delivery Date'])

    if shipment_category != "All":
        shipment_volume_df = shipment_volume_df[shipment_volume_df['Value Category'] == shipment_category]

    shipment_volume_df = shipment_volume_df[
        (shipment_volume_df['Expected Delivery Date'].dt.date >= date_range[0]) &
        (shipment_volume_df['Expected Delivery Date'].dt.date <= date_range[1])
    ]

    shipment_volume_df = shipment_volume_df.groupby("Supplier ID")["Shipment Volume"].sum().reset_index()
    shipment_volume_df.rename(columns={"Shipment Volume": "Total Shipment Volume"}, inplace=True)
    fig_vol = px.bar(
        shipment_volume_df,
        x="Supplier ID",
        y="Total Shipment Volume",
        color="Total Shipment Volume",
        color_continuous_scale="Blues",
        title="Total Shipment Volume per Supplier"
        )
    st.plotly_chart(fig_vol, use_container_width=True)

if supplier_selected != "ALL":

# ----------------------------- SUPPLIER DELIVERY CHART BLOCK -----------------------------
    st.write("## Supplier Delivery Chart")

    # Show average score and flagged state if one supplier is selected
    if supplier_selected != "ALL":
        supplier_data = df[df["Supplier ID"] == supplier_selected]
        avg_score = supplier_data["Supplier Score"].mean()
        flagged = "Yes" if avg_score < 75 else "No"
        st.write(f"**Average Score:** {avg_score:.2f} &nbsp; &nbsp; **Flagged:** {flagged}")

    # View By radiobox
    aggregation_option = st.radio(
        "View By:",
        options=["Per Shipment", "Per Week", "Per Month"],
        index=0,
        horizontal=True
    )

    # Only try to plot if data is available
    if not filtered_df.empty:
        if aggregation_option == "Per Month":
            filtered_df['Expected Period'] = filtered_df['Expected Delivery Date'].dt.to_period('M').dt.to_timestamp()
            filtered_df['Actual Period'] = filtered_df['Actual Delivery Date'].dt.to_period('M').dt.to_timestamp()
            expected_grouped = filtered_df.groupby('Expected Period')['Shipment Volume'].sum().reset_index()
            actual_grouped = filtered_df.groupby('Actual Period')['Shipment Volume'].sum().reset_index()
            x_expected, y_expected = expected_grouped['Expected Period'], expected_grouped['Shipment Volume']
            x_actual, y_actual = actual_grouped['Actual Period'], actual_grouped['Shipment Volume']
        elif aggregation_option == "Per Week":
            filtered_df['Expected Period'] = filtered_df['Expected Delivery Date'].dt.to_period('W').dt.start_time
            filtered_df['Actual Period'] = filtered_df['Actual Delivery Date'].dt.to_period('W').dt.start_time
            expected_grouped = filtered_df.groupby('Expected Period')['Shipment Volume'].sum().reset_index()
            actual_grouped = filtered_df.groupby('Actual Period')['Shipment Volume'].sum().reset_index()
            x_expected, y_expected = expected_grouped['Expected Period'], expected_grouped['Shipment Volume']
            x_actual, y_actual = actual_grouped['Actual Period'], actual_grouped['Shipment Volume']
        else:
            filtered_df = filtered_df.sort_values('Expected Delivery Date')
            x_expected, y_expected = filtered_df['Expected Delivery Date'], filtered_df['Shipment Volume']
            x_actual, y_actual = filtered_df['Actual Delivery Date'], filtered_df['Shipment Volume']

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(x_expected).sort_values(),
            y=y_expected,
            mode='lines+markers',
            name='Expected Delivery',
            line=dict(color='blue', dash='dash'),
            marker=dict(symbol='circle', size=8, color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(x_actual).sort_values(),
            y=y_actual,
            mode='lines+markers',
            name='Actual Delivery',
            line=dict(color='green', dash='solid'),
            marker=dict(symbol='square', size=8, color='green')
        ))

        # Defected and Lost Shipments (for Per Shipment)
        if aggregation_option == "Per Shipment":
            defected_mask = filtered_df['Defected'].astype(str).str.lower() == 'true'
            lost_mask = filtered_df['Shipment Lost'].astype(str).str.lower() == 'true'

            if defected_mask.any():
                defected_x = filtered_df.loc[defected_mask, 'Actual Delivery Date']
                defected_y = filtered_df.loc[defected_mask, 'Shipment Volume']
                fig.add_trace(go.Scatter(
                    x=defected_x,
                    y=defected_y,
                    mode='markers',
                    name='Defected',
                    marker=dict(color='brown', symbol='triangle-up', size=14),
                    showlegend=True
                ))

            if lost_mask.any():
                lost_x = filtered_df.loc[lost_mask, 'Expected Delivery Date']
                lost_y = filtered_df.loc[lost_mask, 'Shipment Volume']
                fig.add_trace(go.Scatter(
                    x=lost_x,
                    y=lost_y,
                    mode='markers',
                    name='Shipment Lost',
                    marker=dict(color='purple', symbol='x', size=14),
                    showlegend=True
                ))

        fig.update_layout(
            title="Expected vs Actual Deliveries",
            xaxis_title="Date",
            yaxis_title="Shipment Volume",
            legend_title="Legend",
            template="plotly_dark",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for the selected filters.")

    # ----------------------------- SUPPLIER OVERVIEW BLOCK -----------------------------
    st.write("## Supplier Overview")
    # Filter supplier overview table by selected supplier and value category
    filtered_overview = overview_df[
                    (overview_df["Supplier ID"] == supplier_selected) &
                    ((overview_df["Value Category"] == shipment_category) if shipment_category != "All" else True)
                ]
    st.dataframe(filtered_overview.reset_index(drop=True), hide_index=True) #to delete index column

else:
    st.info("Select a specific supplier to view delivery chart and supplier overview.")
# -------------------------------- END OF DASHBOARD --------------------------------