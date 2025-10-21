"""
Main Streamlit application for the RAG system.
"""

import streamlit as st
import os
from dotenv import load_dotenv
from rag_system import RAGSystem
from llm_manager import LLMManager

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
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = RAGSystem()
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False

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
            os.environ["OPENAI_API_KEY"] = api_key
            st.session_state.rag_system.llm_manager = LLMManager(api_key=api_key)
            st.success("✅ API Key configured")
        else:
            st.warning("⚠️ API Key required for AI answers")
        
        # Model selection
        model_choice = st.selectbox(
            "LLM Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview", "gpt-5"],
            help="Choose the language model for generating answers"
        )
        
        if api_key:
            st.session_state.rag_system.llm_manager.model = model_choice
        
        st.divider()
        
        # System status
        st.header("📊 System Status")
        status = st.session_state.rag_system.get_system_status()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("PDF Processed", "✅" if status["pdf_processed"] else "❌")
            st.metric("Model Loaded", "✅" if status["model_loaded"] else "❌")
        
        with col2:
            st.metric("LLM Configured", "✅" if status["llm_configured"] else "❌")
            st.metric("Text Chunks", status["num_chunks"])
        
        st.divider()
        
        # Instructions
        st.header("📖 How to Use")
        st.markdown("""
        1. **Upload PDF**: Upload your PDF document
        2. **Wait for Processing**: The system will extract text and create embeddings
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
                with st.spinner("Processing PDF... This may take a few moments."):
                    success = st.session_state.rag_system.process_pdf(uploaded_file)
                    
                    if success:
                        st.session_state.pdf_processed = True
                        st.success("✅ PDF processed successfully! You can now ask questions.")
                    else:
                        st.error("❌ Failed to process PDF. Please try again.")
        
        # Clear session button
        if st.button("🗑️ Clear Session"):
            st.session_state.rag_system = RAGSystem()
            st.session_state.pdf_processed = False
            st.success("Session cleared!")
            st.experimental_rerun()
    
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
                if not st.session_state.rag_system.llm_manager.is_configured():
                    st.error("⚠️ Please configure your OpenAI API key in the sidebar.")
                else:
                    with st.spinner("Generating answer..."):
                        answer = st.session_state.rag_system.ask_question(question, num_chunks)
                        
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
                    st.experimental_rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            Built with Streamlit, LangChain, and OpenAI | RAG System for PDF Q&A
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
