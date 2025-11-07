import os
from dotenv import load_dotenv
import gradio as gr
# ✅ Correct imports for LangChain 1.x
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
# ✅ Fixed import - chains are now in langchain_core
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# =============================
# CONFIGURATION
# =============================
MODEL = "gpt-4o-mini"
DB_NAME = "vector_db_reports"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# =============================
# LOAD HEALTH REPORTS
# =============================
report_folder = "reports"
if not os.path.exists(report_folder):
    raise FileNotFoundError(f"❌ Folder '{report_folder}' not found. Please create it and add .txt reports.")

print(f"🔍 Loading reports from '{report_folder}'...")
loader = DirectoryLoader(report_folder, glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()
print(f"✅ Loaded {len(documents)} reports")

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
print(f"📄 Created {len(chunks)} text chunks")

# =============================
# EMBEDDINGS + VECTORSTORE
# =============================
embeddings = OpenAIEmbeddings(chunk_size=250)

if os.path.exists(DB_NAME):
    print(f"📂 Loading existing Chroma DB '{DB_NAME}'...")
    vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
else:
    print(f"🧠 Creating new Chroma DB in '{DB_NAME}'...")
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=DB_NAME)
    print("✅ Vector DB created!")

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# =============================
# BUILD RETRIEVAL CHAIN
# =============================
llm = ChatOpenAI(model=MODEL, temperature=0.6)

prompt = ChatPromptTemplate.from_template("""
You are an expert AI medical assistant that analyzes text-based blood and health reports.
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
# CHAT FUNCTION
# =============================
def chat(message, history):
    try:
        result = rag_chain.invoke({"input": message})
        answer = result.get("answer", str(result))
        return answer
    except Exception as e:
        return f"❌ Error: {str(e)}"

# =============================
# GRADIO INTERFACE
# =============================
description = """
💬 **AI Health Report Analyzer (RAG)**  
Ask questions about patient blood or health reports stored in the `reports/` folder.  
The assistant will analyze and provide insights and suggestions.
"""

gr.ChatInterface(
    fn=chat,
    title="🩺 Health Report Analyzer",
    description=description,
    theme="default"
).launch(
    share=True,        # 🌍 makes it publicly available
    inbrowser=True   # auto-opens browse
)