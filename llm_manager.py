"""
LLM manager module for generating answers using OpenAI or other language models.
"""

import os
from typing import List, Optional
import streamlit as st
from openai import OpenAI


class LLMManager:
    """Manages language model interactions for generating answers."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the LLM manager.
        
        Args:
            api_key: OpenAI API key
            model: Model name to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None
        
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
    
    def generate_answer(self, query: str, context_chunks: List[tuple], max_tokens: int = 500) -> str:
        """
        Generate an answer based on the query and context chunks.
        
        Args:
            query: User's question
            context_chunks: List of relevant text chunks with similarity scores
            max_tokens: Maximum tokens for the response
            
        Returns:
            Generated answer
        """
        if not self.client:
            return "Error: OpenAI API key not configured. Please set your OPENAI_API_KEY environment variable."
        
        try:
            # Prepare context from chunks
            context = "\n\n".join([chunk[0] for chunk in context_chunks])
            
            # Create the prompt
            prompt = f"""Based on the following context from a PDF document, please answer the user's question. 
            If the answer cannot be found in the context, please say so.

            Context:
            {context}

            Question: {query}

            Answer:"""
            
            # Generate response - use different parameter based on model
            if self.model == "gpt-5":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context. Be concise and accurate."},
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=max_tokens,
                    temperature=0.3
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context. Be concise and accurate."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=0.3
                )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def is_configured(self) -> bool:
        """Check if the LLM manager is properly configured."""
        return self.client is not None
    
    def get_model_info(self) -> str:
        """Get information about the current model."""
        return f"Model: {self.model}, Configured: {self.is_configured()}"
