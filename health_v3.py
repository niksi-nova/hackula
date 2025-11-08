import os
import re
import json
import fitz  # PyMuPDF
import certifi
import bcrypt
import asyncio
import gradio as gr
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from functools import wraps
import nest_asyncio

# Allow nested event loops
nest_asyncio.apply()

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
# CONFIGURATION
# =============================
load_dotenv()
MODEL = "gpt-4o-mini"
DB_NAME = os.getenv("DB_NAME", "medical_vectorstore")
REPORTS_FOLDER = "reports"
MONGO_URI = os.getenv("MONGO_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

os.makedirs(REPORTS_FOLDER, exist_ok=True)

# =============================
# MONGO CONNECTION
# =============================
client = AsyncIOMotorClient(MONGO_URI, tlsCAFile=certifi.where())
db = client[DB_NAME]
users_collection = db["users"]

# =============================
# ASYNC HELPER
# =============================
def run_async(coro):
    """Helper to run async functions in sync context"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        # If loop is already running, create a task
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return loop.run_until_complete(coro)

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

def init_vectorstore():
    if os.path.exists(DB_NAME):
        return Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
    return Chroma(persist_directory=DB_NAME, embedding_function=embeddings)

vectorstore = init_vectorstore()
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
async def signup_async(email, password):
    try:
        if await users_collection.find_one({"email": email}):
            return "❌ User already exists", None
        hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        await users_collection.insert_one({"email": email, "password": hashed.decode("utf-8")})
        return "✅ Account created! Please login.", None
    except Exception as e:
        return f"❌ Error: {str(e)}", None

async def login_async(email, password):
    try:
        user = await users_collection.find_one({"email": email})
        if not user:
            return "❌ No such user", None
        if bcrypt.checkpw(password.encode("utf-8"), user["password"].encode("utf-8")):
            return f"✅ Logged in as {email}", str(user["_id"])
        return "❌ Wrong password", None
    except Exception as e:
        return f"❌ Error: {str(e)}", None

def signup(email, password):
    """Synchronous wrapper for signup"""
    return run_async(signup_async(email, password))

def login(email, password):
    """Synchronous wrapper for login"""
    return run_async(login_async(email, password))


# =============================
# REPORT UPLOAD
# =============================
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

    filename, timestamp = Path(pdf_file.name).stem, datetime.now().strftime("%Y%m%d_%H%M%S")
    content_lines = [
        f"Medical Report - {filename}",
        f"Uploaded by: {user_id}",
        f"Extracted on: {datetime.now()}",
        f"Total Tests: {len(results)}", "",
        "Test Results:", "=" * 50
    ] + [f"{r['test']}: {r['result']}" for r in results]
    content = "\n".join(content_lines)

    txt_filename = f"{filename}_{timestamp}.txt"
    txt_path = os.path.join(REPORTS_FOLDER, txt_filename)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(content)

    doc = Document(page_content=content, metadata={"user_id": user_id, "source": txt_filename})
    chunks = text_splitter.split_documents([doc])
    vectorstore.add_documents(chunks)

    return f"✅ Processed {len(results)} tests\n💾 Saved as {txt_filename}\n👤 Linked to user {user_id}"


# =============================
# CHAT FUNCTION
# =============================
def chat(message, history, user_id):
    if not user_id:
        return "❌ Please log in first."
    
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5, "filter": {"user_id": user_id}})
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
        gr.Markdown("Upload a PDF medical report to extract test results automatically.")
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