cat > README.md << 'EOF'
# 🩺 Health Report Analyzer

An AI-powered medical report analysis system that uses RAG (Retrieval Augmented Generation) to analyze blood test reports and health documents. Upload PDFs, extract test results automatically, and ask natural language questions about your health data.

## ✨ Features

- 📤 **PDF Upload & Extraction**: Automatically extracts test names and results from medical PDFs
- 💬 **Natural Language Chat**: Ask questions about your health reports in plain English
- 🧠 **RAG-Powered**: Uses vector embeddings and retrieval for accurate, context-aware responses
- 🔒 **Local Storage**: All reports stored locally in your `reports/` folder
- 🌐 **Shareable Interface**: Generate public URLs to share with healthcare providers
- 📊 **Multi-Document Support**: Analyze multiple reports and track changes over time

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd health-report-analyzer
```

2. **Create virtual environment**
```bash
python -m venv healthrag
source healthrag/bin/activate  # On Windows: healthrag\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your-openai-api-key-here
```

5. **Run the application**
```bash
python with_ext.py
```

The app will open in your browser at `http://127.0.0.1:7860` and display a public shareable URL.

## 📖 Usage

### Uploading Reports

1. Go to the **"📤 Upload Report"** tab
2. Click "Upload PDF Report" and select your medical report
3. Click **"🔍 Extract & Add to Knowledge Base"**
4. Wait for confirmation that extraction was successful

### Chatting with Your Reports

1. Switch to the **"💬 Chat"** tab
2. Ask questions like:
   - "What are my hemoglobin levels?"
   - "Are there any abnormal values in my reports?"
   - "What dietary changes should I make based on my results?"
   - "Compare my latest report with previous ones"

### Example Questions
```
- What is my current hemoglobin level?
- Show me all abnormal test results
- Has my cholesterol improved since last month?
- What do my liver function tests indicate?
- Give me dietary recommendations based on my results
- Summarize all my blood test reports
```

## 🏗️ Project Structure
```
health-report-analyzer/
├── health_rag.py              # Main application
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (create this)
├── .gitignore                # Git ignore rules
├── README.md                 # This file
├── reports/                  # Extracted reports (auto-created)
│   └── *.txt                # Text versions of uploaded PDFs
└── vector_db_reports/        # Vector database (auto-created)
    └── *.sqlite             # Chroma database files
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key | Yes |

### Application Settings

Edit these in `health_rag.py`:
```python
MODEL = "gpt-4o-mini"              # OpenAI model to use
DB_NAME = "vector_db_reports"      # Vector database directory
REPORTS_FOLDER = "reports"         # Extracted reports directory
```

## 📋 Requirements
```txt
gradio
langchain>=0.3.0
langchain-core>=0.3.0
langchain-community>=0.3.0
langchain-openai>=0.2.0
langchain-chroma>=0.1.0
langchain-text-splitters>=0.3.0
python-dotenv
tiktoken
chromadb
pydantic>=2.0.0,<3.0.0
PyMuPDF>=1.23.0
```

## 🔐 Privacy & Security

- ⚠️ **All medical data stays on your local machine**
- The `.gitignore` excludes `reports/` and `vector_db_reports/` from version control
- Never commit your `.env` file or medical reports to public repositories
- When sharing the Gradio public URL, remember it's accessible to anyone with the link

## 🐛 Troubleshooting

### "No module named 'langchain.chains'"
```bash
pip uninstall langchain langchain-core -y
pip install -r requirements.txt
```

### "API key not found"
Make sure you've created a `.env` file with:
```env
OPENAI_API_KEY=sk-...
```

### "No test results found in PDF"
The PDF format may not be supported. The extractor works best with:
- Standard lab report formats
- Multi-line test results
- Clear test name and value pairs

### Public URL not showing
Remove `server_name` and `server_port` from the `.launch()` call:
```python
demo.launch(share=True, inbrowser=True)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This tool is for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers regarding medical conditions and treatments.

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com/)
- UI powered by [Gradio](https://gradio.app/)
- Vector storage by [Chroma](https://www.trychroma.com/)
- PDF extraction using [PyMuPDF](https://pymupdf.readthedocs.io/)

---

Made with ❤️ for better health data management
EOF
