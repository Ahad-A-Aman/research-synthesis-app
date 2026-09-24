import os
import requests
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(
    page_title="Dual-Source Academic Synthesis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling matching the notebook output design
st.markdown("""
<style>
    .output-container {
        max-width: 900px;
        line-height: 1.8;
        font-size: 15px;
        padding: 24px;
        background-color: #f9f9f9;
        border-left: 5px solid #4A90D9;
        border-radius: 6px;
        margin-top: 20px;
    }
    h2 { color: #2E7D32; margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

st.title("Dual-Source Academic Synthesis Engine")
st.caption("Synthesizing peer-reviewed ERIC research and internal faculty fieldwork via LangChain & Groq")

# Securely bind the Groq API key from Streamlit Cloud Secrets
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("GROQ_API_KEY not found in Streamlit Secrets. Please configure it in the app settings.")
    st.stop()

# Cache the vector store setup so it only runs once per app instance
@st.cache_resource(show_spinner="Building vector databases from ERIC API and internal fieldwork...")
def initialize_retrievers():
    # 1. Fetch external ERIC documents across all 10 research queries
    def fetch_eric_docs(query, num_results=10):
        url = "https://api.ies.ed.gov/eric/"
        params = {
            "search": query,
            "format": "json",
            "rows": num_results,
            "fq": "peerreviewed:T"
        }
        try:
            response = requests.get(url, params=params, timeout=15)
            data = response.json()
        except Exception:
            return []

        documents = []
        for item in data.get("response", {}).get("docs", []):
            if not item.get("description"):
                continue
            content = f"Title: {item.get('title', 'N/A')}\n\nAbstract: {item.get('description', '')}"
            eric_id = item.get("id", "")
            eric_url = f"https://eric.ed.gov/?id={eric_id}" if eric_id else "ERIC database"
            peer = "Yes" if item.get("peerreviewed") == "T" else "No"
            
            metadata = {
                "source": eric_url,
                "source_type": "external",
                "title": item.get("title", "N/A"),
                "author": str(item.get("author", "N/A")),
                "year": str(item.get("publicationdateyear", "N/A")),
                "peer_reviewed": peer
            }
            documents.append(Document(page_content=content, metadata=metadata))
        return documents

    queries = [
        "reading comprehension K-5 elementary",
        "literacy instruction early childhood",
        "phonics decoding elementary students",
        "teacher interview classroom observation",
        "math instruction elementary school",
        "science instruction K-5 students",
        "student engagement classroom learning",
        "formative assessment elementary education",
        "vocabulary development early grades",
        "writing instruction primary school"
    ]

    all_docs = []
    for query in queries:
        all_docs.extend(fetch_eric_docs(query, num_results=10))

    # Deduplicate ERIC documents by title
    seen_titles = set()
    eric_docs = []
    for doc in all_docs:
        if doc.metadata["title"] not in seen_titles:
            seen_titles.add(doc.metadata["title"])
            eric_docs.append(doc)

    # 2. Load internal fieldwork documents
    internal_file_path = "internal_fieldwork.txt"
    if not os.path.exists(internal_file_path):
        st.error(f"Required file '{internal_file_path}' was not found in the repository root.")
        st.stop()

    loader = TextLoader(internal_file_path, encoding='utf-8')
    internal_docs = loader.load()
    for doc in internal_docs:
        doc.metadata["source_type"] = "internal"
        doc.metadata["source"] = "internal_fieldwork.txt"
        doc.metadata["title"] = "Faculty Fieldwork - Classroom Observations, Interviews and Survey"
        doc.metadata["author"] = "Research Team"
        doc.metadata["year"] = "2024"

    # 3. Split documents into chunks
    docs = eric_docs + internal_docs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    external_splits = [s for s in splits if s.metadata.get("source_type") == "external"]
    internal_splits = [s for s in splits if s.metadata.get("source_type") == "internal"]

    # 4. Generate embeddings and initialize vector stores
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    external_vectorstore = Chroma.from_documents(
        documents=external_splits,
        embedding=embeddings,
        collection_name="external_eric"
    )
    
    internal_vectorstore = Chroma.from_documents(
        documents=internal_splits,
        embedding=embeddings,
        collection_name="internal_fieldwork"
    )

    ext_retriever = external_vectorstore.as_retriever(search_kwargs={"k": 5})
    int_retriever = internal_vectorstore.as_retriever(search_kwargs={"k": 3})

    return ext_retriever, int_retriever, len(external_splits), len(internal_splits)

# Initialize retrievers
external_retriever, internal_retriever, ext_count, int_count = initialize_retrievers()

with st.sidebar:
    st.subheader("Database Overview")
    st.write(f"**External ERIC Chunks:** {ext_count}")
    st.write(f"**Internal Fieldwork Chunks:** {int_count}")
    st.divider()
    st.write("**Model:** `openai/gpt-oss-120b` via Groq")
    st.write("**Embeddings:** `all-MiniLM-L6-v2`")

# 5. Define retrieval, formatting, and generation chain
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

prompt = ChatPromptTemplate.from_template("""You are a strict academic research assistant serving education faculty.
Your job is to synthesize findings by combining two distinct types of sources:
1. EXTERNAL SOURCES: Peer-reviewed research from the ERIC academic database
2. INTERNAL SOURCES: The faculty team's own fieldwork including classroom observations, teacher interviews, and surveys

STRICT RULES YOU MUST FOLLOW:
1. ONLY use information explicitly present in the context below. Do not use prior knowledge.
2. ALWAYS clearly distinguish between external research findings and internal fieldwork findings.
3. If the context does not contain enough information, respond only with: "The provided documents do not contain enough information to answer this question."
4. NEVER speculate or generate information beyond what is directly stated in the context.
5. ALWAYS cite the exact source title, author and year for external sources.
6. ALWAYS label internal sources clearly as "Internal Fieldwork" followed by the observation or interview reference.
7. Format your response EXACTLY using the following markdown structure:
## External Research Findings
Write a clear paragraph summarizing what peer-reviewed ERIC research says.
## Internal Fieldwork Findings
Write a clear paragraph summarizing what classroom observations, teacher interviews, and survey data show.
## Synthesis: How They Compare
Write a clear paragraph explaining how the two sources align, complement, or contrast each other.

FORMATTING RULES:
If you find BOTH External and Internal sources, provide a Synthesis paragraph explaining how they compare.
If you ONLY find External sources, summarize them and state: "No internal fieldwork documents were retrieved for this topic."
If you ONLY find Internal sources, summarize them and state: "No external ERIC research was retrieved for this topic."

## Sources Used
**External Sources:**
- [Author(s), Year] - Title
**Internal Sources:**
- [INTERNAL] Description of fieldwork reference (observation/interview/survey)

CONTEXT:
{context}

QUESTION:
{question}

RESPONSE:""")

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

rag_chain = (
    {"context": RunnableLambda(retrieve_from_both), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. Streamlit User Interface
default_question = "How does pre-teaching vocabulary impact reading comprehension for elementary students, particularly for struggling readers?"
user_question = st.text_area("Enter your research question:", value=default_question, height=100)

if st.button("Synthesize Evidence", type="primary"):
    if user_question.strip():
        with st.spinner("Synthesizing external research and internal observations..."):
            try:
                response = rag_chain.invoke(user_question)
                st.markdown(f'<div class="output-container">{response}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error during chain execution: {e}")
    else:
        st.warning("Please enter a research question before executing.")