"""
Startup script for the RAG system.
"""

import os
import sys
import subprocess

def check_env_file():
    """Check if .env file exists and has API key."""
    if not os.path.exists(".env"):
        print("⚠️ .env file not found. Creating template...")
        create_env_template()
        return False
    
    with open(".env", "r") as f:
        content = f.read()
        if "your_openai_api_key_here" in content:
            print("⚠️ Please edit .env file and add your OpenAI API key")
            return False
    
    return True

def create_env_template():
    """Create .env file template."""
    env_content = """# OpenAI API Key (required for LLM functionality)
OPENAI_API_KEY=your_openai_api_key_here

# Optional: You can also use other models
# ANTHROPIC_API_KEY=your_anthropic_api_key_here
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    print("✅ Created .env file template")

def start_streamlit():
    """Start the Streamlit application."""
    print("🚀 Starting RAG System...")
    print("📱 The application will open in your browser")
    print("🔗 URL: http://localhost:8501")
    print("\nPress Ctrl+C to stop the server")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 RAG System stopped")

def main():
    """Main startup function."""
    print("📚 RAG System - PDF Question & Answer")
    print("=" * 50)
    
    # Check environment file
    if not check_env_file():
        print("\n❌ Please configure your API key first:")
        print("1. Edit the .env file")
        print("2. Replace 'your_openai_api_key_here' with your actual API key")
        print("3. Run this script again")
        return
    
    # Start the application
    start_streamlit()

if __name__ == "__main__":
    main()
