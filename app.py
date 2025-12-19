import streamlit as st
import pandas as pd
from transformers import pipeline
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Posta-thon: Smart Grievance System", page_icon="📮", layout="wide")

# --- VIBE LOADER (Dynamic Theme Switching) ---
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"⚠️ Could not find {file_name}. Ensure 'light.css' and 'dark.css' exist!")

# --- 1. LOAD AI MODELS ---
@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3")

with st.spinner("Initializing AI Neural Network..."):
    classifier = load_classifier()

# --- 2. DEFINE LOGIC ---
CANDIDATE_LABELS = ["Delivery Delay", "Staff Behavior", "Damaged/Lost", "General Inquiry", "Financial Services"]

# --- NEW: FULL EMAIL TEMPLATES (Subject + Body) ---
EMAIL_TEMPLATES = {
    "Delivery Delay": """Subject: Update on your consignment delay - India Post
    
Dear Customer,

We sincerely apologize for the delay in your shipment. We have escalated this ticket (ID: #TKT-{id}) to the logistics manager.

Your updated delivery estimate is 24-48 hours. You can track it live on indiapost.gov.in using your consignment number.

Regards,
Central Support Team, India Post""",

    "Staff Behavior": """Subject: Formal acknowledgment of your complaint - India Post
    
Dear Customer,

We take reports of staff misconduct very seriously. This incident has been flagged for immediate disciplinary review by the Regional Superintendent.

We are committed to respectful service. A supervisor will contact you within 4 hours to resolve this.

Regards,
Vigilance Department, India Post""",

    "Damaged/Lost": """Subject: Insurance Claim Process Initiated - India Post
    
Dear Customer,

We regret to hear about the damage. As per policy, we have activated the insurance protocol for your shipment.

Please reply to this email with photos of the damaged package. Our claims team will process your refund (Form-12B) within 3 working days.

Regards,
Claims Division, India Post""",

    "General Inquiry": """Subject: Information Request - India Post Support
    
Dear Customer,

Thank you for contacting India Post. Regarding your inquiry, you can find detailed service guidelines on our official portal.

If you need specific branch details, please visit: https://www.indiapost.gov.in/locate

Regards,
Customer Care, India Post""",

    "Financial Services": """Subject: Investment Services Assistance - India Post
    
Dear Customer,

India Post offers sovereign-backed interest rates on savings schemes. We have forwarded your interest to the nearest Relationship Manager.

They will reach out shortly to guide you through the PLI/RD account opening process.

Regards,
Financial Services Division, India Post"""
}

def get_priority_color(label):
    if label in ["Damaged/Lost", "Staff Behavior"]:
        return "red"
    elif label == "Delivery Delay":
        return "orange"
    return "green"

# --- 3. UI LAYOUT ---

# Sidebar Setup
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c4/India_Post_Logo.svg/1200px-India_Post_Logo.svg.png", width=150)
st.sidebar.title("Admin Panel")

# Theme Toggle
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)
if dark_mode:
    local_css("dark.css")
else:
    local_css("light.css")

st.sidebar.markdown("---")
app_mode = st.sidebar.radio("Select Mode:", ["📝 Live Ticket Analysis", "📂 Batch Processing (CSV)"])
st.sidebar.markdown("---")
st.sidebar.info("System Status: **Online** 🟢")

# Main Title
st.title("📮 AI-Based Complaint Analysis System")
st.markdown("**Problem Statement 3:** Automated classification and prioritization of customer grievances.")

# --- MODE 1: LIVE TICKET ANALYSIS ---
if app_mode == "📝 Live Ticket Analysis":
    st.subheader("Real-Time Complaint Processor")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_input = st.text_area("Enter Customer Complaint:", height=200, placeholder="Paste email or text here...")
        analyze_btn = st.button("Analyze Complaint", type="primary")

    if analyze_btn and user_input:
        start_time = time.time()
        
        # AI PREDICTION
        result = classifier(user_input, CANDIDATE_LABELS)
        top_label = result['labels'][0]
        confidence = result['scores'][0]
        end_time = time.time()
        
        with col2:
            st.markdown("### AI Diagnosis")
            priority_color = get_priority_color(top_label)
            st.markdown(f"**Category:** :{priority_color}[{top_label}]")
            st.markdown(f"**Confidence:** {confidence*100:.2f}%")
            
            if priority_color == "red":
                st.error("⚠️ HIGH PRIORITY TICKET")
            elif priority_color == "orange":
                st.warning("⚠️ MEDIUM PRIORITY TICKET")
            else:
                st.success("✅ STANDARD TICKET")

        st.markdown("---")
        st.subheader("🤖 Generated Auto-Reply")
        # Generate the email preview
        email_draft = EMAIL_TEMPLATES.get(top_label).replace("{id}", "LIVE-001")
        st.text_area("Draft Email:", value=email_draft, height=250)

# --- MODE 2: BATCH PROCESSING ---
elif app_mode == "📂 Batch Processing (CSV)":
    st.subheader("Bulk Data Analytics Dashboard")
    st.markdown("Upload the daily grievance log (CSV) to automatically categorize and generate replies.")
    uploaded_file = st.file_uploader("Upload 'complaints.csv'", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        
        if st.button("Process All Tickets"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Lists to store results
            predicted_labels = []
            draft_replies = []
            
            total_rows = len(df)
            for i, row in df.iterrows():
                text = row['Text'] # Ensure your CSV has a column named 'Text'
                
                # Run AI
                res = classifier(text, CANDIDATE_LABELS)
                top_lbl = res['labels'][0]
                predicted_labels.append(top_lbl)
                
                # Generate Reply
                # We try to use 'Complaint_ID' if it exists, else random ID
                ticket_id = row['Complaint_ID'] if 'Complaint_ID' in df.columns else f"GEN-{i+100}"
                email_body = EMAIL_TEMPLATES.get(top_lbl).replace("{id}", str(ticket_id))
                draft_replies.append(email_body)
                
                # Update progress
                progress = (i + 1) / total_rows
                progress_bar.progress(progress)
                status_text.text(f"Processing Ticket {i+1}/{total_rows}...")
            
            # Add results to DataFrame
            df['AI_Category'] = predicted_labels
            df['Draft_Reply'] = draft_replies
            
            st.success("✅ Processing Complete! All replies generated.")
            
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### 📊 Complaint Distribution")
                st.bar_chart(df['AI_Category'].value_counts())
            with c2:
                st.markdown("### 📋 Preview with Replies")
                # Show the new Draft_Reply column
                st.dataframe(df[['Text', 'AI_Category', 'Draft_Reply']].head(10))
            
            # Download Button
            csv_download = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Report (With Auto-Replies)",
                data=csv_download,
                file_name='processed_complaints_with_replies.csv',
                mime='text/csv',
            )