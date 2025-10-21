"""
Simplified RAG system that works without sentence transformers dependency issues.
"""

import streamlit as st
import os
from dotenv import load_dotenv
import PyPDF2
from openai import OpenAI
import hashlib
import re

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="RAG System - PDF Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'pdf_text' not in st.session_state:
    st.session_state.pdf_text = ""
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks."""
    if not text:
        return []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end < len(text):
            # Try to break at sentence boundary
            sentence_end = text.rfind('.', start, end)
            if sentence_end > start + chunk_size // 2:
                end = sentence_end + 1
            else:
                # Break at word boundary
                word_end = text.rfind(' ', start, end)
                if word_end > start + chunk_size // 2:
                    end = word_end
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

def simple_similarity_search(query, chunks, top_k=5):
    """Simple keyword-based similarity search."""
    query_words = set(query.lower().split())
    scored_chunks = []
    
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        # Calculate Jaccard similarity
        intersection = len(query_words.intersection(chunk_words))
        union = len(query_words.union(chunk_words))
        similarity = intersection / union if union > 0 else 0
        scored_chunks.append((chunk, similarity))
    
    # Sort by similarity and return top chunks
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return scored_chunks[:top_k]

def generate_answer(query, relevant_chunks, api_key, model="gpt-3.5-turbo"):
    """Generate answer using OpenAI."""
    if not api_key:
        return "Please configure your OpenAI API key in the sidebar."
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Prepare context
        context = "\n\n".join([chunk[0] for chunk in relevant_chunks])
        
        prompt = f"""Based on the following context from a PDF document, please answer the user's question. 
        If the answer cannot be found in the context, please say so.

        Context:
        {context}

        Question: {query}

        Answer:"""
        
        # Use different parameter based on model
        if model == "gpt-5":
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context. Be concise and accurate."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=500,
                temperature=0.3
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context. Be concise and accurate."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def main():
    """Main application function."""
    
    # Header
    st.title("📚 RAG System - PDF Question & Answer")
    st.markdown("Upload a PDF document and ask questions about its content using AI-powered retrieval and generation.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Enter your OpenAI API key to enable AI-powered answers"
        )
        
        if api_key:
            st.success("✅ API Key configured")
        else:
            st.warning("⚠️ API Key required for AI answers")
        
        # Model selection
        model_choice = st.selectbox(
            "LLM Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview", "gpt-5"],
            help="Choose the language model for generating answers"
        )
        
        st.divider()
        
        # System status
        st.header("📊 System Status")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("PDF Processed", "✅" if st.session_state.pdf_processed else "❌")
            st.metric("API Key Set", "✅" if api_key else "❌")
        
        with col2:
            st.metric("Text Chunks", len(chunk_text(st.session_state.pdf_text)) if st.session_state.pdf_text else 0)
            st.metric("Model", model_choice)
        
        st.divider()
        
        # Instructions
        st.header("📖 How to Use")
        st.markdown("""
        1. **Upload PDF**: Upload your PDF document
        2. **Wait for Processing**: The system will extract text and create chunks
        3. **Ask Questions**: Type your questions about the document
        4. **Get Answers**: Receive AI-powered answers based on the document content
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 PDF Upload")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF document to analyze"
        )
        
        if uploaded_file is not None:
            # Display file details
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            st.info(f"File size: {uploaded_file.size} bytes")
            
            # Process PDF button
            if st.button("🚀 Process PDF", type="primary"):
                with st.spinner("Processing PDF..."):
                    text = extract_text_from_pdf(uploaded_file)
                    
                    if text:
                        st.session_state.pdf_text = text
                        st.session_state.pdf_processed = True
                        
                        # Show extracted text preview
                        with st.expander("Extracted Text Preview", expanded=False):
                            st.text(text[:1000] + "..." if len(text) > 1000 else text)
                        
                        chunks = chunk_text(text)
                        st.success(f"✅ PDF processed successfully! Created {len(chunks)} text chunks.")
                    else:
                        st.error("❌ Failed to extract text from PDF. Please try again.")
        
        # Clear session button
        if st.button("🗑️ Clear Session"):
            st.session_state.pdf_text = ""
            st.session_state.pdf_processed = False
            st.success("Session cleared!")
            st.rerun()
    
    with col2:
        st.header("❓ Ask Questions")
        
        if not st.session_state.pdf_processed:
            st.info("👆 Please upload and process a PDF file first.")
        else:
            # Question input
            question = st.text_area(
                "Enter your question:",
                placeholder="What is this document about?",
                height=100
            )
            
            # Number of chunks slider
            num_chunks = st.slider(
                "Number of context chunks to retrieve",
                min_value=1,
                max_value=10,
                value=5,
                help="More chunks provide more context but may increase response time"
            )
            
            # Ask question button
            if st.button("🤔 Ask Question", type="primary") and question:
                if not api_key:
                    st.error("⚠️ Please configure your OpenAI API key in the sidebar.")
                else:
                    with st.spinner("Generating answer..."):
                        # Get text chunks
                        chunks = chunk_text(st.session_state.pdf_text)
                        
                        # Find relevant chunks using simple similarity
                        relevant_chunks = simple_similarity_search(question, chunks, num_chunks)
                        
                        # Show retrieved chunks
                        with st.expander("Retrieved Context Chunks", expanded=False):
                            for i, (chunk, score) in enumerate(relevant_chunks):
                                st.write(f"**Chunk {i+1}** (Similarity: {score:.3f}):")
                                st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                                st.write("---")
                        
                        # Generate answer
                        answer = generate_answer(question, relevant_chunks, api_key, model_choice)
                        
                        # Display answer
                        st.markdown("### 💡 Answer:")
                        st.markdown(answer)
            
            # Example questions
            st.markdown("### 💡 Example Questions:")
            example_questions = [
                "What is this document about?",
                "Summarize the main points",
                "What are the key findings?",
                "What conclusions are drawn?",
                "What methodology is used?"
            ]
            
            for example in example_questions:
                if st.button(f"💭 {example}", key=f"example_{example}"):
                    st.session_state.example_question = example
                    st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            Built with Streamlit and OpenAI | Simplified RAG System for PDF Q&A
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
