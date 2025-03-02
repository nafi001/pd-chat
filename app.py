import streamlit as st
import os
from PyPDF2 import PdfReader
from docx import Document
import google.generativeai as genai

# Page configuration
st.set_page_config(page_title="Document Chatbot with Gemini (Free Tier)", layout="wide")

def initialize_gemini_api():
    """Initialize the Gemini API safely"""
    try:
        # Check if API key is in Streamlit secrets
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
            genai.configure(api_key=api_key)
            return genai.GenerativeModel('gemini-pro')  # Correct model name
        else:
            st.warning("⚠️ Gemini API Key not found in secrets!")
            
            # Ask for API key input
            api_key = st.text_input(
                "Enter your Gemini API Key (free to create at https://aistudio.google.com/)",
                type="password"
            )
            
            if api_key:
                genai.configure(api_key=api_key)
                # Save key to session state for current session only
                st.session_state.temp_api_key = api_key
                return genai.GenerativeModel('gemini-pro')  # Correct model name
            else:
                st.info("You need a Google AI Studio account to get a free Gemini API key.")
                st.info("1. Visit https://aistudio.google.com/")
                st.info("2. Create a free account")
                st.info("3. Get your API key from the settings")
                return None
    except Exception as e:
        st.error(f"Error initializing Gemini API: {str(e)}")
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

def find_relevant_chunk(context, question, chunk_size=10000, overlap=1000):
    """Find the most relevant chunk of text for the question to fit in token limits"""
    if not context or len(context) <= chunk_size:
        return context
    
    chunks = []
    start = 0
    while start < len(context):
        end = min(start + chunk_size, len(context))
        chunks.append(context[start:end])
        start += chunk_size - overlap
    
    # Simple keyword matching to find relevant chunks
    question_words = set(question.lower().split())
    chunk_scores = []
    
    for chunk in chunks:
        score = sum(1 for word in question_words if word in chunk.lower())
        chunk_scores.append(score)
    
    best_chunk = chunks[chunk_scores.index(max(chunk_scores))]
    return best_chunk

def generate_response(model, question, context):
    """Generate answer using Gemini"""
    if not model:
        return "API configuration error. Please check your API key."
    
    # Find relevant chunk to fit within model context limits
    relevant_context = find_relevant_chunk(context, question)
    
    prompt = f"""Use the following context to answer the question. If you don't know the answer based on the provided context, say 'I don't know based on the provided information.'
    
    Context: {relevant_context}
    
    Question: {question}
    
    Answer:"""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

def display_gemini_limitations():
    """Display information about Gemini free tier limitations"""
    with st.expander("ℹ️ About Gemini Free Tier Limitations"):
        st.markdown("""
        ### Gemini Free Tier Limitations
        
        **Token Limits:**
        - **What are tokens?** Tokens are pieces of text that the model processes. They're roughly 4 characters or ¾ of a word in English.
        - **Free tier limit:** 60 requests per minute and a monthly quota (typically around 60K input tokens/month)
        - **Context window:** Up to 30K tokens per request (approximately 20-25 pages of text)
        
        **Other Limitations:**
        - Rate limiting may occur during high demand
        - Some advanced features might be restricted
        - The model has a knowledge cutoff date and doesn't know about events after that
        
        **Best Practices:**
        - Provide concise, specific questions
        - For large documents, the app will automatically find the most relevant sections
        - Free tier is suitable for personal projects and learning
        """)

def main():
    st.title("📄 Document Chatbot with Gemini (Free Tier)")
    st.write("Upload a PDF, DOCX, or TXT file and ask questions about its content!")
    
    # Display Gemini limitations
    display_gemini_limitations()
    
    # Initialize the model
    model = None
    if "temp_api_key" in st.session_state:
        genai.configure(api_key=st.session_state.temp_api_key)
        model = genai.GenerativeModel('gemini-pro')  # Correct model name
    else:
        model = initialize_gemini_api()
    
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
                    doc_size_kb = len(context) / 1024
                    token_estimate = len(context) / 4  # Rough estimate of tokens
                    st.success(f"File processed successfully! (~{doc_size_kb:.1f} KB, ~{token_estimate:.0f} tokens)")
                    st.session_state.file_name = uploaded_file.name
    
    # Display chat interface if document is processed
    if st.session_state.get("context"):
        st.write(f"Chatting about: **{st.session_state.get('file_name', 'Document')}**")
        
        # Display token usage estimate
        if len(st.session_state.context) > 30000:
            st.warning(f"This document is large and will be processed in chunks due to token limitations.")
        
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
