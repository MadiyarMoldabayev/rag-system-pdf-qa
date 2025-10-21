"""
Embedding management module for generating and storing document embeddings.
"""

import os
import pickle
import numpy as np
from typing import List, Optional
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss


class EmbeddingManager:
    """Manages document embeddings and vector search."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.index = None
        self.text_chunks = []
        self.embeddings = None
        
    def load_model(self):
        """Load the sentence transformer model."""
        try:
            with st.spinner("Loading embedding model..."):
                self.model = SentenceTransformer(self.model_name)
            st.success(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            st.error(f"Error loading embedding model: {str(e)}")
            raise
    
    def generate_embeddings(self, text_chunks: List[str]) -> np.ndarray:
        """
        Generate embeddings for text chunks.
        
        Args:
            text_chunks: List of text chunks to embed
            
        Returns:
            Array of embeddings
        """
        if not self.model:
            self.load_model()
        
        try:
            with st.spinner("Generating embeddings..."):
                embeddings = self.model.encode(text_chunks, convert_to_numpy=True)
            
            self.text_chunks = text_chunks
            self.embeddings = embeddings
            st.success(f"Generated {len(embeddings)} embeddings")
            
            return embeddings
            
        except Exception as e:
            st.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def build_index(self, embeddings: np.ndarray):
        """
        Build FAISS index for efficient similarity search.
        
        Args:
            embeddings: Array of embeddings to index
        """
        try:
            with st.spinner("Building vector index..."):
                # Create FAISS index
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
                
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(embeddings)
                
                # Add embeddings to index
                self.index.add(embeddings.astype('float32'))
            
            st.success(f"Built vector index with {self.index.ntotal} vectors")
            
        except Exception as e:
            st.error(f"Error building vector index: {str(e)}")
            raise
    
    def search_similar(self, query: str, k: int = 5) -> List[tuple]:
        """
        Search for similar text chunks.
        
        Args:
            query: Query text to search for
            k: Number of similar chunks to return
            
        Returns:
            List of tuples (chunk_text, similarity_score)
        """
        if not self.model or not self.index:
            raise ValueError("Model or index not initialized")
        
        try:
            # Generate embedding for query
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Search for similar vectors
            scores, indices = self.index.search(query_embedding.astype('float32'), k)
            
            # Return results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.text_chunks):
                    results.append((self.text_chunks[idx], float(score)))
            
            return results
            
        except Exception as e:
            st.error(f"Error searching similar chunks: {str(e)}")
            return []
    
    def save_index(self, filepath: str):
        """
        Save the FAISS index and associated data.
        
        Args:
            filepath: Path to save the index
        """
        try:
            # Save FAISS index
            faiss.write_index(self.index, f"{filepath}.index")
            
            # Save associated data
            data = {
                'text_chunks': self.text_chunks,
                'model_name': self.model_name
            }
            
            with open(f"{filepath}.data", 'wb') as f:
                pickle.dump(data, f)
            
            st.success(f"Saved index to {filepath}")
            
        except Exception as e:
            st.error(f"Error saving index: {str(e)}")
    
    def load_index(self, filepath: str):
        """
        Load a previously saved FAISS index.
        
        Args:
            filepath: Path to load the index from
        """
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.index")
            
            # Load associated data
            with open(f"{filepath}.data", 'rb') as f:
                data = pickle.load(f)
            
            self.text_chunks = data['text_chunks']
            self.model_name = data['model_name']
            
            # Load model
            self.load_model()
            
            st.success(f"Loaded index from {filepath}")
            
        except Exception as e:
            st.error(f"Error loading index: {str(e)}")
            raise
