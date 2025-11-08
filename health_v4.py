import os
import re
import json
import fitz  # PyMuPDF
import certifi
import bcrypt
import gradio as gr
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient

# LangChain imports
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


# =============================
# CONFIGURATION
# =============================
load_dotenv()
MODEL = "gpt-4o-mini"
DB_NAME = os.getenv("DB_NAME")
MONGO_URI = os.getenv("MONGO_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# =============================
# MONGO CONNECTION
# =============================
sync_client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
sync_db = sync_client[DB_NAME]
users_collection = sync_db["users"]
reports_collection = sync_db["medical_reports"]
embeddings_collection = sync_db["embeddings"]

# =============================
# PDF EXTRACTOR
# =============================
class MedicalReportExtractor:
    def __init__(self):
        self.skip_keywords = [
            'TEST PARAMETER', 'REFERENCE RANGE', 'RESULT', 'UNIT', 'SAMPLE TYPE',
            'Page', 'Report Status', 'Collected On', 'Reported On', 'Final',
            'Method:', 'Automated', 'Patient Location', 'Flowcytometry',
            'Lab ID', 'UH ID', 'Registered On', 'Age/Gender', 'Electrical Impedence',
            'LABORATORY TEST REPORT', 'HAEMATOLOGY', 'Ref. By', 'Calculated',
            'Processed By', 'End Of Report', 'EDTA', 'Pathologist', 'whole blood',
            'TERMS & CONDITIONS', 'Dr ', 'KMC-', 'Meda Salomi', 'COMPLETE BLOOD COUNT',
            'Male', 'Female', 'Years', 'Name', 'Mr.', 'Mrs.', 'Ms.', 'Differential Leucocyte Count',
            'IP/OP No', 'AKSHAYA NEURO'
        ]
    
    def extract_from_pdf(self, pdf_path: str) -> dict:
        try:
            doc = fitz.open(pdf_path)
            all_results = []
            for page_num in range(len(doc)):
                text = doc[page_num].get_text()
                all_results.extend(self._parse_multiline_format(text))
            doc.close()
            unique_results = self._deduplicate_results(all_results)
            return {"success": True, "results": unique_results}
        except Exception as e:
            return {"success": False, "error": str(e), "results": []}

    def _parse_multiline_format(self, text: str) -> list:
        results, lines = [], [line.strip() for line in text.split('\n')]
        i = 0
        while i < len(lines):
            line = lines[i]
            if not line or self._should_skip_line(line):
                i += 1
                continue
            if self._is_potential_test_name(line):
                test_name, result_value = line, None
                for j in range(i + 1, min(i + 7, len(lines))):
                    next_line = lines[j].strip()
                    if not next_line or any(x in next_line for x in ['Method:', 'Automated', 'Calculated']):
                        continue
                    if self._is_result_value(next_line):
                        result_value = next_line
                        i = j
                        break
                if result_value:
                    results.append({"test": self._clean_test_name(test_name), "result": result_value})
            i += 1
        return results

    def _should_skip_line(self, line): 
        return any(k.lower() in line.lower() for k in self.skip_keywords) or len(line) <= 1 or all(c in '-:/' for c in line)
    
    def _is_potential_test_name(self, line): 
        if len(line) < 3 or not line[0].isupper(): 
            return False
        letters = [c for c in line if c.isalpha()]
        return letters and sum(c.isupper() for c in letters)/len(letters) >= 0.5
    
    def _is_result_value(self, line): 
        return bool(re.match(r'^[\d\.]+$', line))
    
    def _clean_test_name(self, name): 
        return ' '.join(name.split()).rstrip(':').strip()
    
    def _deduplicate_results(self, results):
        seen, unique = set(), []
        for r in results:
            key = (r['test'].lower(), r['result'])
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique


# =============================
# VECTOR STORE + LANGCHAIN
# =============================
embeddings = OpenAIEmbeddings(chunk_size=250)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Initialize MongoDB Atlas Vector Search
vectorstore = MongoDBAtlasVectorSearch(
    collection=embeddings_collection,
    embedding=embeddings,
    index_name="vector_index",
    text_key="text",
    embedding_key="embedding"
)

extractor = MedicalReportExtractor()

llm = ChatOpenAI(model=MODEL, temperature=0.6)
prompt = ChatPromptTemplate.from_template("""
You are an expert AI medical assistant.
Use the provided context to analyze test results and offer clear, evidence-based health insights.
Context:
{context}
Question:
{input}
""")
doc_chain = create_stuff_documents_chain(llm, prompt)

# =============================
# AUTH FUNCTIONS (MongoDB)
# =============================
def signup(email, password):
    try:
        if users_collection.find_one({"email": email}):
            return "❌ User already exists", None
        hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        users_collection.insert_one({"email": email, "password": hashed.decode("utf-8")})
        return "✅ Account created! Please login.", None
    except Exception as e:
        return f"❌ Error: {str(e)}", None

def login(email, password):
    try:
        user = users_collection.find_one({"email": email})
        if not user:
            return "❌ No such user", None
        if bcrypt.checkpw(password.encode("utf-8"), user["password"].encode("utf-8")):
            return f"✅ Logged in as {email}", str(user["_id"])
        return "❌ Wrong password", None
    except Exception as e:
        return f"❌ Error: {str(e)}", None


# =============================
# REPORT UPLOAD (Cloud Storage)
# =============================
def save_report_to_mongodb(user_id, filename, content, results):
    """Save extracted report data to MongoDB"""
    report_doc = {
        "user_id": user_id,
        "filename": filename,
        "content": content,
        "results": results,
        "uploaded_at": datetime.now(),
        "total_tests": len(results)
    }
    result = reports_collection.insert_one(report_doc)
    return str(result.inserted_id)

def process_pdf_upload(pdf_file, user_id):
    if not user_id:
        return "❌ Please log in first."
    
    if pdf_file is None:
        return "❌ Please upload a PDF file."

    extraction = extractor.extract_from_pdf(pdf_file.name)
    if not extraction["success"]:
        return f"❌ Error: {extraction['error']}"
    results = extraction["results"]
    if not results:
        return "⚠️ No test results found."

    filename = Path(pdf_file.name).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build content for vector store
    content_lines = [
        f"Medical Report - {filename}",
        f"Uploaded by: {user_id}",
        f"Extracted on: {datetime.now()}",
        f"Total Tests: {len(results)}", "",
        "Test Results:", "=" * 50
    ] + [f"{r['test']}: {r['result']}" for r in results]
    content = "\n".join(content_lines)

    # Save to MongoDB (cloud storage)
    try:
        report_id = save_report_to_mongodb(user_id, f"{filename}_{timestamp}", content, results)
    except Exception as e:
        return f"❌ Error saving to database: {str(e)}"

    # Create document with metadata for vector store
    doc = Document(
        page_content=content, 
        metadata={
            "user_id": user_id, 
            "source": f"{filename}_{timestamp}",
            "report_id": report_id
        }
    )
    
    # Split and add to MongoDB vector store
    try:
        chunks = text_splitter.split_documents([doc])
        vectorstore.add_documents(chunks)
    except Exception as e:
        return f"❌ Error adding to vector store: {str(e)}"

    return f"✅ Processed {len(results)} tests\n☁️ Saved to cloud database\n📊 Report ID: {report_id}\n👤 Linked to user {user_id}"


# =============================
# CHAT FUNCTION
# =============================
def chat(message, history, user_id):
    if not user_id:
        return "❌ Please log in first."
    
    try:
        # Filter by user_id to only retrieve their reports
        # MongoDB Atlas uses pre_filter instead of filter
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 5,
                "pre_filter": {"user_id": user_id}
            }
        )
        chain = create_retrieval_chain(retriever, doc_chain)
        result = chain.invoke({"input": message})
        return result.get("answer", "⚠️ No relevant data found.")
    except Exception as e:
        return f"❌ Error: {str(e)}"


# =============================
# GRADIO INTERFACE
# =============================
def handle_auth(email, password, action):
    if not email or not password:
        return "❌ Please provide both email and password", None
    
    if action == "Signup":
        msg, uid = signup(email, password)
        return msg, uid
    else:
        msg, uid = login(email, password)
        return msg, uid


with gr.Blocks(title="Medical Report Analyzer") as demo:
    session_state = gr.State()
    
    with gr.Tab("🔐 Login / Signup"):
        gr.Markdown("## Welcome to Medical Report Analyzer")
        email = gr.Textbox(label="Email", placeholder="Enter your email")
        password = gr.Textbox(label="Password", type="password", placeholder="Enter your password")
        action = gr.Radio(["Login", "Signup"], label="Action", value="Login")
        auth_btn = gr.Button("Submit", variant="primary")
        auth_status = gr.Textbox(label="Status", lines=2, interactive=False)
        
        auth_btn.click(
            handle_auth, 
            inputs=[email, password, action], 
            outputs=[auth_status, session_state]
        )

    with gr.Tab("📤 Upload Report"):
        gr.Markdown("## Upload Your Medical Report")
        gr.Markdown("Upload a PDF medical report to extract test results automatically. All data is stored securely in the cloud.")
        pdf_input = gr.File(label="Upload Medical Report (PDF)", file_types=[".pdf"], type="filepath")
        upload_btn = gr.Button("🔍 Extract & Save", variant="primary")
        upload_status = gr.Textbox(label="Upload Status", lines=5, interactive=False)
        
        upload_btn.click(
            process_pdf_upload, 
            inputs=[pdf_input, session_state], 
            outputs=upload_status
        )

    with gr.Tab("💬 Chat"):
        gr.Markdown("## Ask Questions About Your Health Reports")
        gr.Markdown("Chat with AI to understand your medical test results and get health insights.")
        
        def chat_wrapper(message, history, user_id):
            return chat(message, history, user_id)
        
        chatbot = gr.ChatInterface(
            fn=chat_wrapper,
            additional_inputs=[session_state],
            type="messages",
            examples=[
                ["What are my latest blood test results?"],
                ["Are any of my test values abnormal?"],
                ["Explain my hemoglobin levels"],
                ["What do these results mean for my health?"]
            ]
        )

if __name__ == "__main__":
    demo.launch(share=True, inbrowser=True)