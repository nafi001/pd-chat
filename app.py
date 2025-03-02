import streamlit as st
import os
from PyPDF2 import PdfReader
from docx import Document
import google.generativeai as genai

# Page configuration
st.set_page_config(page_title="Document Chatbot with Gemini", layout="wide")

def initialize_api():
    """Initialize the Gemini API safely"""
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-pro')
    except KeyError:
        st.error("⚠️ Gemini API Key not found! Please add it to your Streamlit secrets.")
        st.info("For local development: Create a .streamlit/secrets.toml file with GEMINI_API_KEY = 'your-key'")
        st.info("For Streamlit Cloud: Add the secret in App Settings > Secrets")
        return None

def extract_text(file):
    """Extract text from PDF, DOCX, or TXT files"""
    try:
        if file.type == "application/pdf":
            reader = PdfReader(file)
            text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
            if not text.strip():
                st.warning("The PDF appears to be scanned or doesn't contain extractable text.")
            return text
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(file)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        elif file.type == "text/plain":
            text = file.read().decode("utf-8", errors="replace")
            return text
        else:
            st.error(f"Unsupported file type: {file.type}")
            return None
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return None

def generate_response(model, question, context):
    """Generate answer using Gemini"""
    if not model:
        return "API configuration error. Please check your API key."
    
    # Limit context to Gemini's token limit (approximately 30k tokens)
    max_context_chars = 30000
    if len(context) > max_context_chars:
        context = context[:max_context_chars]
        st.info(f"The document was truncated to fit the model's token limit.")
    
    prompt = f"""Use the following context to answer the question. If you don't know the answer based on the provided context, say 'I don't know'.
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

def main():
    st.title("📄 Document Chatbot with Gemini")
    st.write("Upload a PDF, DOCX, or TXT file and ask questions about its content!")
    
    # Initialize the model
    model = initialize_api()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize document context
    if "context" not in st.session_state:
        st.session_state.context = None
    
    # Process uploaded file
    if uploaded_file:
        if st.button("Process Document") or st.session_state.context is None:
            with st.spinner("Processing document..."):
                context = extract_text(uploaded_file)
                if context:
                    st.session_state.context = context
                    st.success(f"File processed successfully! ({len(context)} characters)")
                    st.session_state.file_name = uploaded_file.name
    
    # Display chat interface if document is processed
    if st.session_state.get("context"):
        st.write(f"Chatting about: **{st.session_state.get('file_name', 'Document')}**")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Handle user input
        if question := st.chat_input("Ask about the document"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": question})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(question)
            
            # Generate and display assistant response
            with st.spinner("Thinking..."):
                answer = generate_response(model, question, st.session_state.context)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(answer)
    
    # Add option to clear chat history
    if st.session_state.get("context") and st.session_state.messages:
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.experimental_rerun()

if __name__ == "__main__":
    main()
