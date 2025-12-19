import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from transformers import pipeline
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="India Post AI Dashboard", page_icon="🇮🇳", layout="wide")

# --- CSS LOADING ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load the file
local_css("style.css")

# --- SETUP AI & DATA ---
@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3")

with st.spinner("Initializing Digital India AI Core..."):
    classifier = load_classifier()

LABELS = ["Parcel Delivery Delay", "Staff Misconduct", "Damaged Article", "Financial Services", "General Inquiry", "Spam"]

# Session State for Page Navigation
if 'page' not in st.session_state:
    st.session_state.page = "Dashboard"

if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["ID", "Customer", "Complaint", "Category", "Priority", "Status", "Sentiment", "Date", "Draft_Reply"])

df = st.session_state.data

# --- SIDEBAR NAVIGATION ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c4/India_Post_Logo.svg/1200px-India_Post_Logo.svg.png", width=120)
st.sidebar.markdown("<h3 style='text-align: center; color: #333;'>Seva & Suvidha</h3>", unsafe_allow_html=True)
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# 1. MAIN NAVIGATION (Now Tricolor Buttons)
if st.sidebar.button("📊  Dashboard", use_container_width=True):
    st.session_state.page = "Dashboard"
    st.rerun()

if st.sidebar.button("📈  Analytics", use_container_width=True):
    st.session_state.page = "Analytics"
    st.rerun()

if st.sidebar.button("🤖  AI Config", use_container_width=True):
    st.session_state.page = "AI Config"
    st.rerun()

if st.sidebar.button("⚙️  Settings", use_container_width=True):
    st.session_state.page = "Settings"
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("GRIEVANCE INBOX")

# 2. COMPLAINTS LIST (Text Links)
with st.sidebar.expander("📂 All Complaints", expanded=True):
    if df.empty:
        st.caption("No active grievances.")
    else:
        filter_cat = st.selectbox("Category", ["All"] + LABELS, index=0, key="sb_filter")
        sidebar_df = df if filter_cat == "All" else df[df['Category'] == filter_cat]
        
        for i, row in sidebar_df.head(20).iterrows():
            label = f"{row['ID']} - {row['Customer'].split()[0]}"
            if st.button(label, key=f"btn_{row['ID']}", use_container_width=True):
                st.session_state.page = "Complaints"
                st.session_state.selected_ticket = row
                st.rerun()

# --- PAGE LOGIC ---

# 1. DASHBOARD
if st.session_state.page == "Dashboard":
    st.title("Overview")
    if df.empty:
        st.warning("⚠️ System Idle. Please upload a CSV file to begin analysis.")
        
    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Complaints", len(df))
    m2.metric("AI Resolved", int(len(df)*0.7))
    m3.metric("Pending", int(len(df)*0.1))
    m4.metric("Avg Response", "2.4h")
    
    st.divider()
    
    # Upload & Manual Entry
    st.subheader("📝 Complaint Processing Unit")
    tab1, tab2 = st.tabs(["✍️ Manual Entry", "📂 Bulk CSV Upload"])
    
    with tab1:
        c1, c2 = st.columns([2, 1])
        with c1:
            manual_text = st.text_area("Paste unstructured text/email here:", height=150)
            if st.button("Analyze Now"):
                if manual_text:
                    with st.spinner("AI analyzing..."):
                        time.sleep(1)
                        res = classifier(manual_text, LABELS)
                        st.success(f"**Category:** {res['labels'][0]}")
                        st.info(f"**Confidence:** {res['scores'][0]*100:.2f}%")
        with c2:
            st.info("ℹ️ **Gov AI Tip:** Detects spam, urgency, and routing automatically.")

    with tab2:
        st.markdown("Upload your daily `complaints.csv`.")
        uploaded_file = st.file_uploader("Drag & Drop CSV File", type=["csv"])
        
        if uploaded_file:
            if st.button("Process Bulk File"):
                raw_df = pd.read_csv(uploaded_file)
                rename_map = {"Complaint_ID": "ID", "Customer_Name": "Customer", "Text": "Complaint", "Date": "Date"}
                raw_df.rename(columns=rename_map, inplace=True)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                categories, priorities, sentiments, statuses, drafts = [], [], [], [], []
                total_rows = len(raw_df)
                
                for index, row in raw_df.iterrows():
                    status_text.text(f"Processing Ticket {index+1}/{total_rows}...")
                    progress_bar.progress((index + 1) / total_rows)
                    time.sleep(0.05)
                    
                    text = row['Complaint']
                    res = classifier(text, LABELS)
                    top_label = res['labels'][0]
                    categories.append(top_label)
                    
                    if top_label in ["Damaged Article", "Staff Misconduct"]:
                        priorities.append("High"); sentiments.append("Negative")
                    else:
                        priorities.append("Medium"); sentiments.append("Neutral")
                        
                    statuses.append("Open")
                    drafts.append(f"Dear {row['Customer']},\n\nWe received your complaint regarding {top_label}...")
                
                raw_df['Category'] = categories
                raw_df['Priority'] = priorities
                raw_df['Sentiment'] = sentiments
                raw_df['Status'] = statuses
                raw_df['Draft_Reply'] = drafts
                
                st.session_state.data = raw_df
                st.success("✅ Analysis Complete! Dashboard Updated.")
                time.sleep(1)
                st.rerun()

# 2. ANALYTICS
elif st.session_state.page == "Analytics":
    st.title("Analytics")
    if not df.empty:
        chart_data = df['Category'].value_counts().reset_index()
        chart_data.columns = ['Category', 'Count']
        
        bar_chart = alt.Chart(chart_data).mark_bar(
            cornerRadius=6,
            color='#FF9933'
        ).encode(
            x=alt.X('Category', axis=alt.Axis(labelAngle=0, grid=False)),
            y=alt.Y('Count', axis=alt.Axis(grid=False)),
        ).properties(background='transparent')
        
        st.altair_chart(bar_chart, use_container_width=True)
    else:
        st.info("No data available.")

# 3. SETTINGS
elif st.session_state.page == "Settings":
    st.title("Settings")
    st.text_input("Admin Email", "postmaster@indiapost.gov.in")

# 4. COMPLAINTS
elif st.session_state.page == "Complaints":
    if 'selected_ticket' in st.session_state:
        row = st.session_state.selected_ticket
        st.button("← Back to Dashboard", on_click=lambda: st.session_state.update(page="Dashboard"))
        
        st.title(f"Ticket: {row['ID']}")
        
        c1, c2 = st.columns([2,1])
        with c1:
            st.container().markdown(f"**Message:**\n\n{row['Complaint']}")
            st.text_area("Draft Reply", row.get('Draft_Reply', 'Generated reply...'), height=150)
            if st.button("Send Reply"):
                st.success("Sent via SpeedPost Digital!")
        with c2:
            st.info(f"Customer: {row['Customer']}")
            st.warning(f"Category: {row['Category']}")
            s_color = "red" if row['Sentiment'] == "Negative" else ("green" if row['Sentiment'] == "Positive" else "orange")
            st.markdown(f"**Sentiment:** :{s_color}[{row['Sentiment']}]")
    else:
        st.info("Select a complaint from the sidebar.")
