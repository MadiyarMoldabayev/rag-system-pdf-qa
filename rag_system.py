"""
Main RAG (Retrieval-Augmented Generation) system that combines all components.
"""

import os
from typing import List, Optional
import streamlit as st
from pdf_processor import PDFProcessor
from embedding_manager import EmbeddingManager
from llm_manager import LLMManager


class RAGSystem:
    """Main RAG system that orchestrates PDF processing, embedding, and LLM generation."""
    
    def __init__(self):
        """Initialize the RAG system with all components."""
        self.pdf_processor = PDFProcessor()
        self.embedding_manager = EmbeddingManager()
        self.llm_manager = LLMManager()
        self.is_initialized = False
    
    def initialize(self):
        """Initialize the embedding model."""
        if not self.is_initialized:
            self.embedding_manager.load_model()
            self.is_initialized = True
    
    def process_pdf(self, pdf_file) -> bool:
        """
        Process uploaded PDF file and create embeddings.
        
        Args:
            pdf_file: Uploaded PDF file
            
        Returns:
            True if processing was successful, False otherwise
        """
        try:
            # Initialize if not done yet
            self.initialize()
            
            # Extract text from PDF
            text = self.pdf_processor.extract_text_from_pdf(pdf_file)
            if not text:
                st.error("No text could be extracted from the PDF.")
                return False
            
            # Show extracted text preview
            with st.expander("Extracted Text Preview", expanded=False):
                st.text(text[:1000] + "..." if len(text) > 1000 else text)
            
            # Create text chunks
            chunks = self.pdf_processor.get_text_chunks(text)
            st.info(f"Created {len(chunks)} text chunks from the PDF.")
            
            # Generate embeddings
            embeddings = self.embedding_manager.generate_embeddings(chunks)
            
            # Build vector index
            self.embedding_manager.build_index(embeddings)
            
            return True
            
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return False
    
    def ask_question(self, question: str, num_chunks: int = 5) -> str:
        """
        Ask a question and get an answer based on the processed PDF.
        
        Args:
            question: User's question
            num_chunks: Number of relevant chunks to retrieve
            
        Returns:
            Generated answer
        """
        if not self.embedding_manager.index:
            return "Please upload and process a PDF file first."
        
        if not self.llm_manager.is_configured():
            return "OpenAI API key not configured. Please set your OPENAI_API_KEY environment variable."
        
        try:
            # Search for relevant chunks
            similar_chunks = self.embedding_manager.search_similar(question, k=num_chunks)
            
            if not similar_chunks:
                return "No relevant information found in the document."
            
            # Show retrieved chunks (optional)
            with st.expander("Retrieved Context Chunks", expanded=False):
                for i, (chunk, score) in enumerate(similar_chunks):
                    st.write(f"**Chunk {i+1}** (Similarity: {score:.3f}):")
                    st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                    st.write("---")
            
            # Generate answer
            answer = self.llm_manager.generate_answer(question, similar_chunks)
            
            return answer
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def get_system_status(self) -> dict:
        """Get the current status of the RAG system."""
        return {
            "pdf_processed": bool(self.embedding_manager.index),
            "model_loaded": self.is_initialized,
            "llm_configured": self.llm_manager.is_configured(),
            "num_chunks": len(self.embedding_manager.text_chunks) if self.embedding_manager.text_chunks else 0,
            "model_info": self.llm_manager.get_model_info()
        }
