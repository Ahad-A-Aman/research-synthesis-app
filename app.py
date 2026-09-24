import streamlit as st
import os
import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Academic Synthesis Engine", layout="wide")
st.title("Dual-Source Academic Synthesis")

# Fetch the Groq API key securely from Streamlit secrets
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# Cache the databases so they only build once when the app boots
@st.cache_resource(show_spinner="Initializing vector databases...")
def build_databases():
    # 1. Load External Data (ERIC)
    def fetch_eric_docs(query, num_results=10):
        url = "https://api.ies.ed.gov/eric/"
        params = {"search": query, "format": "json", "rows": num_results, "fq": "peerreviewed:T"}
        try:
            response = requests.get(url, params=params)
            data = response.json()
        except:
            return []
        
        documents = []
        for item in data.get("response", {}).get("docs", []):
            if not item.get("description"): continue
            content = f"Title: {item.get('title', 'N/A')}\n\nAbstract: {item.get('description', '')}"
            metadata = {
                "source_type": "external",
                "title": item.get("title", "N/A"),
                "author": str(item.get("author", "N/A")),
                "year": str(item.get("publicationdateyear", "N/A"))
            }
            documents.append(Document(page_content=content, metadata=metadata))
        return documents

    # Reduced query list to speed up the initial cloud server boot time
    queries = [
        "reading comprehension K-5 elementary",
        "literacy instruction early childhood",
        "student engagement classroom learning"
    ] 
    
    all_docs = []
    for query in queries:
        all_docs.extend(fetch_eric_docs(query, num_results=10))

    # 2. Load Internal Data
    loader = TextLoader("internal_fieldwork.txt", encoding='utf-8')
    internal_docs = loader.load()
    for doc in internal_docs:
        doc.metadata["source_type"] = "internal"
        doc.metadata["title"] = "Faculty Fieldwork"

    # 3. Split and Embed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs + internal_docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    external_splits = [s for s in splits if s.metadata.get("source_type") == "external"]
    internal_splits = [s for s in splits if s.metadata.get("source_type") == "internal"]
    
    ext_store = Chroma.from_documents(external_splits, embeddings, collection_name="external_eric")
    int_store = Chroma.from_documents(internal_splits, embeddings, collection_name="internal_fieldwork")
    
    return ext_store.as_retriever(search_kwargs={"k": 5}), int_store.as_retriever(search_kwargs={"k": 3})

external_retriever, internal_retriever = build_databases()

# 4. Initialize LLM and RAG Chain
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

def format_docs(docs):
    formatted = []
    for doc in docs:
        source_type = doc.metadata.get("source_type", "unknown").upper()
        title = doc.metadata.get("title", "Unknown")
        formatted.append(f"[{source_type} SOURCE]\nTitle: {title}\nContent: {doc.page_content}")
    return "\n\n--\n\n".join(formatted)

def retrieve_from_both(question):
    ext_docs = external_retriever.invoke(question)
    int_docs = internal_retriever.invoke(question)
    return format_docs(ext_docs + int_docs)

prompt = ChatPromptTemplate.from_template("""
You are a strict academic research assistant serving education faculty.
Your job is to synthesize findings by combining two distinct types of sources:
1. EXTERNAL SOURCES: Peer-reviewed research from the ERIC academic database
2. INTERNAL SOURCES: The faculty team's own fieldwork

STRICT RULES YOU MUST FOLLOW:
1. ONLY use information explicitly present in the context below. Do not use prior knowledge.
2. ALWAYS clearly distinguish between external research findings and internal fieldwork findings.
3. If the context does not contain enough information, respond only with: "The provided documents do not contain enough inf"
4. NEVER speculate or generate information beyond what is directly stated in the context.
5. ALWAYS cite the exact source title, author and year for external sources.
6. ALWAYS label Internal sources clearly as "Internal Fieldwork" followed by the observation or interview reference.
7. Format your response EXACTLY using the following markdown structure:
## External Research Findings
Write a clear paragraph summarizing what peer-reviewed ERIC research says.
## Internal Fieldwork Findings
Write a clear paragraph summarizing what classroom observations, teacher interviews, and survey data show.
## Synthesis: How They Compare
Write a clear paragraph explaining how the two sources align, complement, or contrast each other.

CONTEXT:
{context}

QUESTION:
{question}
""")

rag_chain = (
    {"context": RunnableLambda(retrieve_from_both), "question": RunnablePassthrough()}
    | prompt 
    | llm 
    | StrOutputParser()
)

# 5. User Interface
question = st.text_input("Enter your research question:")

if st.button("Synthesize"):
    if question:
        with st.spinner("Analyzing ERIC and internal fieldwork documents..."):
            response = rag_chain.invoke(question)
            st.markdown(response)
    else:
        st.warning("Please enter a question first.")