import os
from dotenv import load_dotenv
import gradio as gr
import fitz  # PyMuPDF
import re
import json
from pathlib import Path
from datetime import datetime
# LangChain imports
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# =============================
# PDF EXTRACTOR CLASS
# =============================
class MedicalReportExtractor:
    """Extract test results from medical PDFs"""
    
    def __init__(self):
        self.skip_keywords = [
            'TEST PARAMETER', 'REFERENCE RANGE', 'RESULT', 'UNIT', 'SAMPLE TYPE',
            'Page', 'Report Status', 'Collected On', 'Reported On', 'Final',
            'Method:', 'Automated', 'Patient Location', 'Flowcytometry',
            'Lab ID', 'UH ID', 'Registered On', 'Age/Gender', 'Electrical Impedence',
            'LABORATORY TEST REPORT', 'HAEMATOLOGY', 'Ref. By', 'Calculated',
            'Processed By', 'End Of Report', 'EDTA', 'Pathologist', 'whole blood',
            'TERMS & CONDITIONS', 'Dr ', 'KMC-', 'Meda Salomi',
            'COMPLETE BLOOD COUNT', 'Male', 'Female', 'Years', 'Name', 'Mr.', 'Mrs.', 'Ms.',
            'Differential Leucocyte Count', 'IP/OP No', 'AKSHAYA NEURO'
        ]
    
    def extract_from_pdf(self, pdf_path: str) -> dict:
        """Extract test results from PDF"""
        try:
            doc = fitz.open(pdf_path)
            all_results = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                page_results = self._parse_multiline_format(text)
                all_results.extend(page_results)
            
            doc.close()
            unique_results = self._deduplicate_results(all_results)
            
            return {
                'success': True,
                'total_tests': len(unique_results),
                'results': unique_results
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'results': []
            }
    
    def _parse_multiline_format(self, text: str) -> list:
        """Parse multi-line format test results"""
        results = []
        lines = [line.strip() for line in text.split('\n')]
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            if not line or self._should_skip_line(line):
                i += 1
                continue
            
            if self._is_potential_test_name(line):
                test_name = line
                result_value = None
                
                for j in range(i + 1, min(i + 7, len(lines))):
                    if j >= len(lines):
                        break
                    
                    next_line = lines[j].strip()
                    if not next_line:
                        continue
                    
                    if any(skip in next_line for skip in ['Method:', 'Automated', 'Calculated']):
                        continue
                    
                    if self._is_result_value(next_line):
                        result_value = next_line
                        i = j
                        break
                
                if result_value:
                    cleaned_name = self._clean_test_name(test_name)
                    results.append({
                        'test': cleaned_name,
                        'result': result_value
                    })
            
            i += 1
        
        return results
    
    def _is_potential_test_name(self, line: str) -> bool:
        """Check if line looks like a test name"""
        if len(line) < 3 or not line[0].isupper():
            return False
        if self._should_skip_line(line):
            return False
        
        letters = [c for c in line if c.isalpha()]
        if not letters:
            return False
        
        uppercase_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        return uppercase_ratio >= 0.5
    
    def _is_result_value(self, line: str) -> bool:
        """Check if line is a result value"""
        pattern = r'^[\d\.]+$'
        return bool(re.match(pattern, line))
    
    def _should_skip_line(self, line: str) -> bool:
        """Check if line should be skipped"""
        for keyword in self.skip_keywords:
            if keyword.lower() in line.lower():
                return True
        if len(line) <= 1 or all(c in '-:/' for c in line):
            return True
        return False
    
    def _clean_test_name(self, name: str) -> str:
        """Clean test name"""
        name = ' '.join(name.split())
        return name.rstrip(':').strip()
    
    def _deduplicate_results(self, results: list) -> list:
        """Remove duplicate results"""
        seen = set()
        unique = []
        for item in results:
            key = (item['test'].lower(), item['result'])
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique

# =============================
# CONFIGURATION
# =============================
MODEL = "gpt-4o-mini"
DB_NAME = "vector_db_reports"
REPORTS_FOLDER = "reports"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Create reports folder if it doesn't exist
os.makedirs(REPORTS_FOLDER, exist_ok=True)

# =============================
# INITIALIZE COMPONENTS
# =============================
extractor = MedicalReportExtractor()
embeddings = OpenAIEmbeddings(chunk_size=250)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Initialize or load vector store
def init_vectorstore():
    """Initialize vector store"""
    if os.path.exists(DB_NAME):
        print(f"📂 Loading existing Chroma DB '{DB_NAME}'...")
        return Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
    else:
        print(f"🧠 Creating new Chroma DB in '{DB_NAME}'...")
        # Create empty vectorstore
        return Chroma(persist_directory=DB_NAME, embedding_function=embeddings)

vectorstore = init_vectorstore()

# Load existing reports
def load_existing_reports():
    """Load existing text reports into vectorstore"""
    if os.path.exists(REPORTS_FOLDER):
        loader = DirectoryLoader(REPORTS_FOLDER, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()
        if documents:
            chunks = text_splitter.split_documents(documents)
            vectorstore.add_documents(chunks)
            print(f"✅ Loaded {len(documents)} existing reports ({len(chunks)} chunks)")

load_existing_reports()

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# =============================
# BUILD RETRIEVAL CHAIN
# =============================
llm = ChatOpenAI(model=MODEL, temperature=0.6)

prompt = ChatPromptTemplate.from_template("""
You are an expert AI medical assistant that analyzes blood and health reports.
Use the provided context to infer patient health, identify abnormalities, and give
specific advice for diet or lifestyle improvements.

Context:
{context}

Question:
{input}

If you cannot find relevant data, say so politely.
""")

doc_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, doc_chain)

# =============================
# UPLOAD AND PROCESS PDF
# =============================
def process_pdf_upload(pdf_file):
    """Process uploaded PDF and add to vector store"""
    if pdf_file is None:
        return "❌ No file uploaded"
    
    try:
        # Extract data from PDF
        extraction_result = extractor.extract_from_pdf(pdf_file.name)
        
        if not extraction_result['success']:
            return f"❌ Extraction failed: {extraction_result.get('error', 'Unknown error')}"
        
        results = extraction_result['results']
        
        if not results:
            return "⚠️ No test results found in the PDF"
        
        # Create formatted text content
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = Path(pdf_file.name).stem
        
        content_lines = [
            f"Medical Report - {filename}",
            f"Extracted on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Tests: {len(results)}",
            "",
            "Test Results:",
            "=" * 50
        ]
        
        for item in results:
            content_lines.append(f"{item['test']}: {item['result']}")
        
        content = "\n".join(content_lines)
        
        # Save to reports folder
        txt_filename = f"{filename}_{timestamp}.txt"
        txt_path = os.path.join(REPORTS_FOLDER, txt_filename)
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Add to vector store
        doc = Document(page_content=content, metadata={"source": txt_filename})
        chunks = text_splitter.split_documents([doc])
        vectorstore.add_documents(chunks)
        
        return f"✅ Successfully processed!\n\n📄 Extracted {len(results)} test results\n💾 Saved as: {txt_filename}\n🧠 Added to knowledge base"
        
    except Exception as e:
        return f"❌ Error processing PDF: {str(e)}"

# =============================
# CHAT FUNCTION
# =============================
def chat(message, history):
    """Handle chat messages"""
    try:
        result = rag_chain.invoke({"input": message})
        answer = result.get("answer", str(result))
        return answer
    except Exception as e:
        return f"❌ Error: {str(e)}"

# =============================
# GRADIO INTERFACE
# =============================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🩺 Health Report Analyzer")
    gr.Markdown("Upload medical PDFs or chat about uploaded reports")
    
    with gr.Tab("💬 Chat"):
        gr.Markdown("Ask questions about uploaded health reports")
        chatbot = gr.ChatInterface(
            fn=chat,
            examples=[
                "What are my hemoglobin levels?",
                "Are there any abnormal values in my reports?",
                "What dietary changes should I make based on my results?",
                "Summarize all my test results"
            ]
        )
    
    with gr.Tab("📤 Upload Report"):
        gr.Markdown("### Upload a Medical Report PDF")
        gr.Markdown("The system will extract test results and add them to the knowledge base")
        
        with gr.Row():
            pdf_input = gr.File(
                label="Upload PDF Report",
                file_types=[".pdf"],
                type="filepath"
            )
        
        upload_btn = gr.Button("🔍 Extract & Add to Knowledge Base", variant="primary")
        upload_output = gr.Textbox(label="Status", lines=5)
        
        upload_btn.click(
            fn=process_pdf_upload,
            inputs=pdf_input,
            outputs=upload_output
        )
        
        gr.Markdown("### 📋 Tips:")
        gr.Markdown("""
        - Upload blood test reports, lab results, or health check-ups
        - PDFs should contain test names and results
        - After upload, switch to the Chat tab to ask questions
        - The system automatically detects and extracts medical test data
        """)

demo.launch(
    share=True,
    inbrowser=True
)