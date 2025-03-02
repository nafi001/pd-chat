import streamlit as st
import os
from PyPDF2 import PdfReader
from docx import Document
import google.generativeai as genai

# Configure Gemini
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-pro')

def extract_text(file):
    """Extract text from PDF, DOCX, or TXT files"""
    if file.type == "application/pdf":
        reader = PdfReader(file)
        text = " ".join([page.extract_text() for page in reader.pages])
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
    elif file.type == "text/plain":
        text = file.read().decode()
    else:
        raise ValueError("Unsupported file type")
    return text

def generate_response(question, context):
    """Generate answer using Gemini"""
    prompt = f"""Use the following context to answer the question. If you don't know the answer, say 'I don't know'.
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
    
    response = model.generate_content(prompt)
    return response.text
import streamlit as st
st.write("API Key (partial):", st.secrets["GEMINI_API_KEY"][:4] + "****")
# Streamlit UI
st.title("📄 Document Chatbot with Gemini")
st.write("Upload a PDF, DOCX, or TXT file and ask questions!")

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])

if uploaded_file:
    try:
        context = extract_text(uploaded_file)
        st.success("File processed successfully!")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        if question := st.chat_input("Ask about the document"):
            st.session_state.messages.append({"role": "user", "content": question})
            
            with st.spinner("Thinking..."):
                answer = generate_response(question, context[:30000])  # Gemini's token limit
                
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            with st.chat_message("user"):
                st.markdown(question)
            
            with st.chat_message("assistant"):
                st.markdown(answer)
                
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
