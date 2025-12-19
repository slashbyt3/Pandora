import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from transformers import pipeline
import time
import random

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="India Post AI Dashboard", page_icon="📮", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    /* 1. Main Background */
    .stApp { background-color: #F8F9FA; color: #333; }
    
    /* 2. Sidebar Styling to mimic the reference image */
    section[data-testid="stSidebar"] { 
        background-color: #FFFFFF; 
        border-right: 1px solid #E0E0E0;
    }
    
    /* 3. Cards */
    div[data-testid="stMetric"], div[data-testid="stContainer"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* 4. Buttons (India Post Red) */
    div.stButton > button {
        background-color: #D32F2F; color: white; border: none;
        border-radius: 4px; padding: 8px 16px;
    }
    div.stButton > button:hover { background-color: #B71C1C; }
    
    </style>
    """, unsafe_allow_html=True)

# --- 1. SETUP AI MODELS & DATA ---
@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3")

with st.spinner("Initializing AI Core..."):
    classifier = load_classifier()

LABELS = ["Delivery Delay", "Staff Behavior", "Damaged/Lost", "Financial Services", "General Inquiry", "Fake/Spam Complaint"]

# --- MOCK DATA GENERATOR ---
if 'data' not in st.session_state:
    data = []
    names = ["Amit Sharma", "Priya Singh", "Rahul Verma", "Sneha Gupta", "Mohd Faiz"]
    statuses = ["Open", "In Progress", "Resolved", "Pending Review"]
    sentiments = ["Negative", "Neutral", "Positive"]
    
    for i in range(50):
        category = random.choice(LABELS)
        data.append({
            "ID": f"TKT-{1000+i}",
            "Customer": random.choice(names),
            "Complaint": f"Sample complaint text regarding {category}...",
            "Category": category,
            "Priority": "High" if category in ["Damaged/Lost", "Staff Behavior"] else "Medium",
            "Status": random.choice(statuses),
            "Sentiment": random.choice(sentiments),
            "Date": "2025-12-19"
        })
    st.session_state.data = pd.DataFrame(data)

df = st.session_state.data

# --- FUNCTION: COMPLAINT DETAIL POPUP ---
@st.dialog("Ticket Analysis Details")
def show_complaint_details(row):
    st.markdown(f"### Ticket ID: {row['ID']}")
    st.caption(f"Filed on {row['Date']} by {row['Customer']}")
    st.divider()
    
    c1, c2 = st.columns(2)
    with c1:
        st.info(f"**Category:** {row['Category']}")
        p_color = "red" if row['Priority'] == "High" else "orange"
        st.markdown(f"**Priority:** :{p_color}[{row['Priority']}]")
    with c2:
        st.info(f"**Status:** {row['Status']}")
        st.markdown(f"**Sentiment:** {row['Sentiment']}")

    st.markdown("#### Complaint Text")
    st.text_area("Customer Message", row['Complaint'], disabled=True)
    
    st.markdown("#### AI Recommended Reply")
    st.text_area("Draft Email", f"Dear {row['Customer']},\n\nWe apologize for the issue regarding {row['Category']}. Your ticket {row['ID']} is prioritized.\n\nRegards,\nIndia Post", height=150)
    
    col_a, col_b = st.columns(2)
    if col_a.button("✅ Approve & Send"):
        st.success("Reply Sent!")
        time.sleep(1)
        st.rerun()
    if col_b.button("🚫 Mark as Spam"):
        st.error("Ticket Closed as Spam")
        time.sleep(1)
        st.rerun()

# --- 2. SIDEBAR NAVIGATION ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c4/India_Post_Logo.svg/1200px-India_Post_Logo.svg.png", width=120)
st.sidebar.title("India Post AI")

# Navigation Menu
nav = st.sidebar.radio("Main Menu", ["Dashboard", "Analytics", "Settings", "AI Config"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Grievance Handling**")

# Interactive Complaints List in Sidebar
with st.sidebar.expander("📂 All Complaints", expanded=True):
    # Quick filters in sidebar
    filter_cat = st.selectbox("Filter Category", ["All"] + LABELS, index=0)
    
    # Filter logic
    sidebar_df = df if filter_cat == "All" else df[df['Category'] == filter_cat]
    
    # List of buttons
    for i, row in sidebar_df.head(10).iterrows(): # Limiting to 10 for performance
        if st.button(f"{row['ID']} - {row['Customer']}", key=f"btn_{row['ID']}", use_container_width=True):
            show_complaint_details(row)

# --- PAGE 1: DASHBOARD ---
if nav == "Dashboard":
    st.title("Overview")
    
    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Complaints", len(df), "+12%")
    m2.metric("AI Resolved", int(len(df)*0.72), "72% Rate")
    m3.metric("Pending", int(len(df)*0.1), "-5%")
    m4.metric("Avg Response", "2.4h", "Stable")
    
    st.divider()

    # Split View: Manual Processor + CSV Upload
    st.subheader("📝 Complaint Processing Unit")
    
    tab1, tab2 = st.tabs(["✍️ Manual Entry", "📂 Bulk CSV Upload"])
    
    with tab1:
        c1, c2 = st.columns([2, 1])
        with c1:
            manual_text = st.text_area("Paste unstructured text/email here:", height=150)
            if st.button("Analyze Now"):
                if manual_text:
                    res = classifier(manual_text, LABELS)
                    st.success(f"**Predicted Category:** {res['labels'][0]}")
                    st.info(f"**Confidence Score:** {res['scores'][0]*100:.2f}%")
        with c2:
            st.info("ℹ️ **AI Tip:** Paste full email body. The model detects spam and urgency automatically.")

    with tab2:
        st.markdown("Upload your daily `complaints.csv` for batch processing.")
        uploaded_file = st.file_uploader("Drag & Drop CSV File", type=["csv"])
        
        if uploaded_file:
            uploaded_df = pd.read_csv(uploaded_file)
            st.dataframe(uploaded_df.head(3))
            
            if st.button("Process Bulk File"):
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress.progress(i+1)
                st.success("✅ Batch Analysis Complete! Reports generated.")
                st.download_button("📥 Download Report", data=uploaded_df.to_csv(), file_name="processed_report.csv")

    st.divider()
    
    # Analytics Section (Bottom)
    st.subheader("📊 Live Trends")
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.caption("Complaint Volume by Category")
        chart_data = df['Category'].value_counts().reset_index()
        chart_data.columns = ['Category', 'Count']
        bar_chart = alt.Chart(chart_data).mark_bar(color='#D32F2F').encode(
            x='Category', y='Count', tooltip=['Category', 'Count']
        ).properties(height=300)
        st.altair_chart(bar_chart, use_container_width=True)

    with chart_col2:
        st.caption("Sentiment Distribution")
        sent_data = df['Sentiment'].value_counts().reset_index()
        sent_data.columns = ['Sentiment', 'Count']
        pie = alt.Chart(sent_data).mark_arc(innerRadius=60).encode(
            theta='Count',
            color=alt.Color('Sentiment', scale=alt.Scale(domain=['Positive', 'Neutral', 'Negative'], range=['#2E7D32', '#F9A825', '#C62828']))
        )
        st.altair_chart(pie, use_container_width=True)

# --- OTHER PAGES (Placeholders) ---
elif nav == "Analytics":
    st.title("Deep Analytics")
    st.info("Advanced historical data and trend forecasting module.")

elif nav == "Settings":
    st.title("Admin Settings")
    st.text_input("System Admin Email", value="admin@indiapost.gov.in")

elif nav == "AI Config":
    st.title("Model Configuration")
    st.slider("Confidence Threshold", 0.0, 1.0, 0.75)