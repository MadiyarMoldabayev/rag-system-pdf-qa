"""
Test script to verify that all dependencies are properly installed.
"""

def test_imports():
    """Test if all required packages can be imported."""
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import PyPDF2
        print("✅ PyPDF2 imported successfully")
    except ImportError as e:
        print(f"❌ PyPDF2 import failed: {e}")
        return False
    
    try:
        import sentence_transformers
        print("✅ Sentence Transformers imported successfully")
    except ImportError as e:
        print(f"❌ Sentence Transformers import failed: {e}")
        return False
    
    try:
        import faiss
        print("✅ FAISS imported successfully")
    except ImportError as e:
        print(f"❌ FAISS import failed: {e}")
        return False
    
    try:
        from openai import OpenAI
        print("✅ OpenAI imported successfully")
    except ImportError as e:
        print(f"❌ OpenAI import failed: {e}")
        return False
    
    try:
        from dotenv import load_dotenv
        print("✅ Python-dotenv imported successfully")
    except ImportError as e:
        print(f"❌ Python-dotenv import failed: {e}")
        return False
    
    return True

def test_local_modules():
    """Test if local modules can be imported."""
    try:
        from pdf_processor import PDFProcessor
        print("✅ PDFProcessor imported successfully")
    except ImportError as e:
        print(f"❌ PDFProcessor import failed: {e}")
        return False
    
    try:
        from embedding_manager import EmbeddingManager
        print("✅ EmbeddingManager imported successfully")
    except ImportError as e:
        print(f"❌ EmbeddingManager import failed: {e}")
        return False
    
    try:
        from llm_manager import LLMManager
        print("✅ LLMManager imported successfully")
    except ImportError as e:
        print(f"❌ LLMManager import failed: {e}")
        return False
    
    try:
        from rag_system import RAGSystem
        print("✅ RAGSystem imported successfully")
    except ImportError as e:
        print(f"❌ RAGSystem import failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🧪 Testing RAG System Installation...")
    print("=" * 50)
    
    print("\n📦 Testing external dependencies...")
    deps_ok = test_imports()
    
    print("\n🏠 Testing local modules...")
    modules_ok = test_local_modules()
    
    print("\n" + "=" * 50)
    if deps_ok and modules_ok:
        print("🎉 All tests passed! Your RAG system is ready to use.")
        print("\nTo start the application, run:")
        print("streamlit run app.py")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("\nTo install missing dependencies, run:")
        print("pip install -r requirements.txt")
