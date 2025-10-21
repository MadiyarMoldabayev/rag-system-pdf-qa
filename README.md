# RAG System - PDF Question & Answer

A Retrieval-Augmented Generation (RAG) system that allows you to upload PDF documents and ask questions about their content using AI-powered natural language processing.

🔗 **GitHub Repository**: [https://github.com/MadiyarMoldabayev/rag-system-pdf-qa](https://github.com/MadiyarMoldabayev/rag-system-pdf-qa)

## Features

- 📄 **PDF Text Extraction**: Automatically extracts text from uploaded PDF files
- 🔍 **Semantic Search**: Uses sentence transformers to find relevant document chunks
- 🤖 **AI-Powered Answers**: Generates contextual answers using OpenAI's language models
- 🎨 **User-Friendly Interface**: Clean Streamlit web interface
- ⚡ **Fast Retrieval**: FAISS vector database for efficient similarity search

## Installation

1. **Clone or download this project**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up OpenAI API Key**:
   - Get your API key from [OpenAI](https://platform.openai.com/api-keys)
   - Create a `.env` file in the project directory:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```
   - Or set the environment variable:
     ```bash
     export OPENAI_API_KEY=your_api_key_here
     ```

## Usage

1. **Run the application**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

3. **Upload a PDF**:
   - Click "Browse files" and select your PDF document
   - Click "Process PDF" to extract text and create embeddings

4. **Ask questions**:
   - Type your question in the text area
   - Click "Ask Question" to get an AI-powered answer

## How It Works

1. **Text Extraction**: The system extracts text from your PDF using PyPDF2
2. **Text Chunking**: The extracted text is split into overlapping chunks for better retrieval
3. **Embedding Generation**: Each chunk is converted to a vector using sentence transformers
4. **Vector Storage**: Embeddings are stored in a FAISS index for fast similarity search
5. **Question Processing**: Your question is converted to a vector and matched against document chunks
6. **Answer Generation**: The most relevant chunks are sent to an LLM to generate a contextual answer

## Configuration

### Models Used

- **Embedding Model**: `all-MiniLM-L6-v2` (sentence-transformers)
- **LLM Models**: GPT-3.5-turbo, GPT-4, or GPT-4-turbo-preview (configurable)

### Customization

You can modify the following parameters in the code:

- **Chunk Size**: Default 1000 characters (in `pdf_processor.py`)
- **Overlap**: Default 200 characters between chunks
- **Number of Retrieved Chunks**: Default 5 (adjustable in the UI)
- **Embedding Model**: Change in `embedding_manager.py`

## File Structure

```
rag_system/
├── app.py                 # Main Streamlit application
├── rag_system.py         # Core RAG system orchestration
├── pdf_processor.py      # PDF text extraction and chunking
├── embedding_manager.py  # Embedding generation and vector search
├── llm_manager.py        # Language model integration
├── requirements.txt      # Python dependencies
├── env_example.txt       # Environment variables example
└── README.md            # This file
```

## Requirements

- Python 3.8+
- OpenAI API key
- Internet connection for downloading models and API calls

## Troubleshooting

### Common Issues

1. **"OpenAI API key not configured"**
   - Make sure you've set your API key in the `.env` file or environment variables

2. **"Error extracting text from PDF"**
   - Ensure the PDF is not password-protected
   - Try with a different PDF file
   - Some PDFs with images may not extract text properly

3. **Slow processing**
   - Large PDFs may take time to process
   - The embedding model downloads on first use (one-time)

4. **"No relevant information found"**
   - Try rephrasing your question
   - Increase the number of context chunks
   - Check if your question is related to the document content

### Performance Tips

- Use smaller chunk sizes for more precise retrieval
- Increase the number of retrieved chunks for broader context
- Process PDFs with clear, readable text for best results

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this RAG system.
