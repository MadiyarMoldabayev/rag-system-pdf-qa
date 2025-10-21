"""
Setup script for the RAG system.
"""

import os
import sys
import subprocess

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def create_env_file():
    """Create .env file template."""
    env_content = """# OpenAI API Key (required for LLM functionality)
OPENAI_API_KEY=your_openai_api_key_here

# Optional: You can also use other models
# ANTHROPIC_API_KEY=your_anthropic_api_key_here
"""
    
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write(env_content)
        print("✅ Created .env file template")
        print("⚠️ Please edit .env file and add your OpenAI API key")
    else:
        print("✅ .env file already exists")

def test_installation():
    """Test if the installation is working."""
    print("\n🧪 Testing installation...")
    try:
        # Test basic imports
        import streamlit
        import PyPDF2
        import openai
        import faiss
        print("✅ Core dependencies imported successfully")
        
        # Test local modules
        from pdf_processor import PDFProcessor
        from llm_manager import LLMManager
        print("✅ Local modules imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 RAG System Setup")
    print("=" * 50)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed at dependency installation")
        return False
    
    # Create environment file
    create_env_file()
    
    # Test installation
    if test_installation():
        print("\n🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Edit .env file and add your OpenAI API key")
        print("2. Run: streamlit run app.py")
        print("3. Open your browser and start using the RAG system!")
        return True
    else:
        print("\n⚠️ Setup completed with warnings")
        print("Some dependencies may need manual installation")
        return False

if __name__ == "__main__":
    main()
