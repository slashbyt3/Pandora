import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from transformers import pipeline
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="India Post Workspace", page_icon="📬", layout="wide")

# --- LOAD CSS ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# --- SETUP AI ---
@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3")

with st.spinner("Loading Workspace..."):
    classifier = load_classifier()

LABELS = ["Parcel Delivery Delay", "Staff Misconduct", "Damaged Article", "Financial Services", "General Inquiry", "Spam"]

# --- SESSION STATE ---
if 'page' not in st.session_state: st.session_state.page = "Dashboard"
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["ID", "Customer", "Complaint", "Category", "Priority", "Status", "Sentiment", "Date", "Draft_Reply"])

df = st.session_state.data

# --- SIDEBAR (NOTION STYLE) ---
with st.sidebar:
    # 1. Brand Header (HTML from Friend's code)
    st.markdown("""
        <div style="padding: 10px 0px; display:flex; align-items:center; gap:8px; margin-bottom:10px; cursor:pointer;">
            <div style="width:22px; height:22px; background:#D32F2F; color:white; border-radius:4px; display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:bold;">IP</div>
            <div style="font-weight:600; color:#37352F; font-size:14px;">India Post Workspace</div>
        </div>
    """, unsafe_allow_html=True)

    # 2. Navigation (Styled as links via CSS)
    if st.button("⊞  Dashboard", use_container_width=True): st.session_state.page = "Dashboard"; st.rerun()
    if st.button("📈  Analytics", use_container_width=True): st.session_state.page = "Analytics"; st.rerun()
    if st.button("⚙️  Configuration", use_container_width=True): st.session_state.page = "Config"; st.rerun()
    if st.button("📥  Complaints DB", use_container_width=True): st.session_state.page = "Complaints"; st.rerun()

    # 3. Active Tickets List
    st.markdown("""
        <div style="padding: 24px 12px 8px 12px; font-size:11px; font-weight:600; color:#787774; display:flex; justify-content:space-between; letter-spacing:0.5px;">
            <span>ACTIVE TICKETS</span>
            <span style="cursor:pointer;">+</span>
        </div>
    """, unsafe_allow_html=True)

    # Your Functionality: Sidebar List Logic
    if df.empty:
        st.caption("No active tickets.")
    else:
        # We use a container so scrolling looks nice
        with st.container():
            # Loop through data (Your Logic)
            for i, row in df.head(15).iterrows():
                # Notion-style row look
                icon = "📄"
                if "Damaged" in row['Category']: icon = "📦"
                if "Financial" in row['Category']: icon = "💰"
                
                label = f"{icon} {row['Customer'].split()[0]}'s Ticket"
                
                # The Click Functionality
                if st.button(label, key=f"btn_{row['ID']}", use_container_width=True):
                    st.session_state.selected_ticket = row
                    st.session_state.page = "TicketDetail"
                    st.rerun()

# --- MAIN CONTENT AREA ---

# 1. Notion Cover Image (From Friend's HTML)
st.markdown('<div class="notion-cover"></div>', unsafe_allow_html=True)

# 2. Header Block
st.markdown("""
    <div style="margin-bottom:40px;">
        <div style="font-size:72px; margin-bottom:0px; margin-left:-5px;">📬</div>
        <h1 style="font-size:40px; font-weight:700; margin:0; color:#37352F;">Complaint Management System</h1>
        <div style="display:flex; gap:16px; color:#787774; font-size:14px; margin-top:12px; border-bottom:1px solid #E9E9E7; padding-bottom:24px;">
            <span><i class="far fa-user"></i> Admin Workspace</span>
            <span><i class="far fa-clock"></i> Updated just now</span>
            <span><i class="far fa-comment"></i> 12 Comments</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- VIEW: DASHBOARD ---
if st.session_state.page == "Dashboard":
    
    # 3. KPI Cards (Your Data + Friend's HTML Structure)
    st.subheader("Overview")
    
    k1, k2, k3, k4 = st.columns(4)
    
    # Helper to create HTML Card
    def card(title, value, sub, color="gray"):
        color_map = {"green": "#DBEDDB", "red": "#FFDCE0", "blue": "#D3E5EF", "gray": "#F1F1EF"}
        text_map = {"green": "#4C8B58", "red": "#D1576B", "blue": "#2383E2", "gray": "#787774"}
        
        return f"""
        <div style="border:1px solid #E9E9E7; border-radius:4px; padding:16px; background:white; height:100%;">
            <div style="color:#787774; font-size:14px; display:flex; align-items:center; gap:6px;">
                {title}
            </div>
            <div style="font-size:32px; font-weight:600; color:#37352F; margin-top:10px;">{value}</div>
            <div style="margin-top:10px;">
                <span style="background:{color_map[color]}; color:{text_map[color]}; padding:2px 6px; border-radius:4px; font-size:12px;">{sub}</span>
            </div>
        </div>
        """

    # Calculating Metrics (Your Logic)
    total = len(df)
    resolved = int(total * 0.72)
    pending = int(total * 0.1)
    
    with k1: st.markdown(card("Total Complaints", total, "12% vs last mo", "green"), unsafe_allow_html=True)
    with k2: st.markdown(card("AI Resolved", resolved, "72% Auto-rate", "blue"), unsafe_allow_html=True)
    with k3: st.markdown(card("Pending", pending, "Needs Action", "red"), unsafe_allow_html=True)
    with k4: st.markdown(card("Avg Response", "2.4h", "Stable", "gray"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Split Layout (Analytics + Manual Entry)
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.write("### 📊 Live Analytics")
        if not df.empty:
            # Clean Altair Chart
            chart_data = df['Category'].value_counts().reset_index()
            chart_data.columns = ['Category', 'Count']
            chart = alt.Chart(chart_data).mark_bar(cornerRadius=3, color='#D32F2F').encode(
                x=alt.X('Category', axis=alt.Axis(labelAngle=0, grid=False)),
                y=alt.Y('Count', axis=alt.Axis(grid=False)),
                tooltip=['Category', 'Count']
            ).properties(height=280)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No data. Upload CSV to see trends.")

    with c2:
        st.write("### 🤖 Manual AI Analysis")
        # Styled like a Notion Callout
        with st.container(border=True):
            st.caption("Paste unstructured text/email here:")
            manual_text = st.text_area("Input", height=120, label_visibility="collapsed", placeholder="e.g. 'Speed Post delayed...'")
            
            if st.button("Analyze Now", use_container_width=True):
                if manual_text:
                    # Your AI Logic
                    res = classifier(manual_text, LABELS)
                    cat = res['labels'][0]
                    conf = int(res['scores'][0]*100)
                    
                    # Result Display (Tags)
                    st.markdown(f"""
                        <div style="margin-top:10px; padding:10px; background:#F7F7F5; border-radius:4px;">
                            <div style="font-size:12px; color:#787774;">DETECTED INTENT</div>
                            <div style="margin-top:4px;">
                                <span class="n-tag tag-blue">{cat}</span>
                                <span class="n-tag tag-gray">{conf}% Confidence</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

    # 5. Complaint Processing Unit (Functionality: CSV Upload)
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("### 📥 Complaint Processing Unit")
    
    # Custom Notion-like file uploader area
    with st.container(border=True):
        st.markdown("**Bulk Upload (.csv)**")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        
        if uploaded_file:
            if st.button("Process Bulk File"):
                # --- YOUR EXACT LOGIC ---
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
                    time.sleep(0.02) # Visual
                    
                    text = row['Complaint']
                    res = classifier(text, LABELS)
                    top_cat = res['labels'][0]
                    categories.append(top_cat)
                    
                    if top_cat in ["Damaged Article", "Staff Misconduct"]:
                        priorities.append("High"); sentiments.append("Negative")
                    else:
                        priorities.append("Medium"); sentiments.append("Neutral")
                    
                    statuses.append("Open")
                    drafts.append(f"Dear {row['Customer']}, regarding {top_cat}...")
                
                raw_df['Category'] = categories; raw_df['Priority'] = priorities
                raw_df['Sentiment'] = sentiments; raw_df['Status'] = statuses
                raw_df['Draft_Reply'] = drafts
                
                st.session_state.data = raw_df
                st.success("✅ Database Updated Successfully")
                time.sleep(1); st.rerun()

# --- VIEW: TICKET DETAIL (The Functionality triggered by sidebar) ---
elif st.session_state.page == "TicketDetail":
    if 'selected_ticket' in st.session_state:
        row = st.session_state.selected_ticket
        
        # Navigation
        st.button("← Back to Overview", on_click=lambda: st.session_state.update(page="Dashboard"))
        
        # Header
        st.markdown(f"## Ticket: {row['ID']}")
        st.markdown(f"**Customer:** {row['Customer']} &nbsp; | &nbsp; **Date:** {row['Date']}")
        st.divider()
        
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.markdown("#### 💬 Customer Message")
            st.info(row['Complaint'])
            
            st.markdown("#### ✍️ Draft Reply")
            st.text_area("Edit Draft", row.get('Draft_Reply', ''), height=200)
            
            if st.button("Send Response", type="primary"):
                st.success("Message dispatched successfully.")

        with c2:
            # Notion Style Property List
            st.markdown("#### 🧠 AI Analysis")
            
            # Helper for property row
            def prop_row(label, value, color="tag-gray"):
                return f"""
                <div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #E9E9E7; align-items:center;">
                    <span style="color:#787774; font-size:14px;">{label}</span>
                    <span class="n-tag {color}">{value}</span>
                </div>
                """
            
            # Logic for colors
            p_color = "tag-red" if row['Priority'] == "High" else "tag-yellow"
            s_color = "tag-red" if row['Sentiment'] == "Negative" else "tag-green"
            
            st.markdown(prop_row("Category", row['Category'], "tag-blue"), unsafe_allow_html=True)
            st.markdown(prop_row("Priority", row['Priority'], p_color), unsafe_allow_html=True)
            st.markdown(prop_row("Sentiment", row['Sentiment'], s_color), unsafe_allow_html=True)
            st.markdown(prop_row("Status", row['Status'], "tag-gray"), unsafe_allow_html=True)

# --- VIEW: ANALYTICS ---
elif st.session_state.page == "Analytics":
    st.subheader("📈 Analytics & Reports")
    if not df.empty:
        st.bar_chart(df['Category'].value_counts())
    else:
        st.info("Upload data in Dashboard first.")

# --- VIEW: CONFIG ---
elif st.session_state.page == "Config":
    st.subheader("⚙️ Settings")
    st.text_input("System Admin", "admin@indiapost.gov.in")