import os
import requests
import time
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
        color: #1a1a1a; /* Forces text to be dark grey */
        border-left: 5px solid #4A90D9;
        border-radius: 6px;
        margin-top: 20px;
    }
    .output-container h2 { 
        color: #2E7D32 !important; 
        margin-top: 20px; 
    }
    .output-container p, .output-container li {
        color: #1a1a1a !important;
    }
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
@st.cache_resource(show_spinner="Loading pre-built vector databases...")
def load_databases():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    external_vectorstore = Chroma(
        collection_name="external_eric",
        embedding_function=embeddings,
        persist_directory="./chroma_db_backup"
    )
    
    internal_vectorstore = Chroma(
        collection_name="internal_fieldwork",
        embedding_function=embeddings,
        persist_directory="./chroma_db_backup"
    )

    ext_retriever = external_vectorstore.as_retriever(search_kwargs={"k": 5})
    int_retriever = internal_vectorstore.as_retriever(search_kwargs={"k": 3})

    # Hardcoding the chunk counts based on your Colab output to keep the sidebar UI working
    return ext_retriever, int_retriever, 281, 39

# Initialize retrievers
external_retriever, internal_retriever, ext_count, int_count = load_databases()

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
        author = doc.metadata.get("author", "Unknown Author")
        year = doc.metadata.get("year", "Unknown Year")
        
        # Now the AI will receive the exact author and year alongside the text
        formatted.append(f"[{source_type} SOURCE]\nTitle: {title}\nAuthor: {author}\nYear: {year}\nContent: {doc.page_content}")
    
    return "\n\n--\n\n".join(formatted)

def retrieve_from_both(question):
    ext_docs = external_retriever.invoke(question)
    int_docs = internal_retriever.invoke(question)
    return format_docs(ext_docs + int_docs)

prompt = ChatPromptTemplate.from_template("""You are an academic research assistant serving education faculty.
Your job is to synthesize findings by combining two distinct types of sources:
1. EXTERNAL SOURCES: Peer-reviewed research from the ERIC academic database
2. INTERNAL SOURCES: The faculty team's own fieldwork including observations, interviews, and surveys

STRICT RULES YOU MUST FOLLOW:
1. ONLY use information explicitly present in the context below. Do not hallucinate or use outside knowledge.
2. ALWAYS clearly distinguish between external research findings and internal fieldwork findings.
3. Synthesize whatever relevant information is available in the context, even if it only partially answers the question.
4. If the context is 100% unrelated to the question, respond ONLY with: "The provided documents do not contain enough information to answer this question."
5. Format your response EXACTLY using the following markdown structure. You MUST place a blank line after every heading:

## External Research Findings

[Summarize relevant peer-reviewed ERIC research here. If none is relevant, write: "No external ERIC research was retrieved for this topic."]

## Internal Fieldwork Findings

[Summarize relevant classroom observations, teacher interviews, and survey data here. If none is relevant, write: "No internal fieldwork documents were retrieved for this topic."]

## Synthesis: How They Compare

[If BOTH sources are present, explain how they align, complement, or contrast. If only one source type is present, omit this section.]

## Sources Used

[You MUST list all sources referenced in your synthesis here. Do not hallucinate sources. Use the exact metadata provided in the context.]
**External Sources:**
- **[Title]** by [Author] ([Year])

**Internal Sources:**
- **Internal Fieldwork**: [Brief description of the specific observation, interview, or survey cited]

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
default_question = "How do I improve vocabulary of k-5 learners?"
user_question = st.text_area("Enter your research question:", value=default_question, height=100)

if st.button("Synthesize Evidence", type="primary"):
    if user_question.strip():
        with st.spinner("Synthesizing external research and internal observations..."):
            
            # 1. Intercept the context before sending it to the AI
            raw_context = retrieve_from_both(user_question)
            

            # 3. Run the standard generation chain
            max_retries = 4
            for attempt in range(max_retries):
                try:
                    response = rag_chain.invoke(user_question)
                    response = response.replace("## External Research Findings", "## External Research Findings\n\n")
                    st.markdown(f'<div class="output-container">\n\n{response}\n\n</div>', unsafe_allow_html=True)
                    
                    # Add this verification block immediately after the response
                    with st.expander("📚 Verify Sources (View Raw Data)"):
                        st.info("Cross-check the AI's citations against the exact documents retrieved from the database below.")
                        st.text(raw_context)
                        
                    break  # Stop the loop immediately if it succeeds
                
                except Exception as e:
                    if attempt < max_retries - 1:
                        st.toast(f"API hiccup (Attempt {attempt + 1}). Retrying...")
                        import time
                        time.sleep(2)
                    else:
                        st.error(f"Error during chain execution: {e}")
    else:
        st.warning("Please enter a research question before executing.")
