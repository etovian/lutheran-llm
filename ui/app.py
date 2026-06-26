import streamlit as st
import logging
from config.settings import Settings
from database.connection import get_engine, check_connection
from pipeline.orchestrator import run_orchestrator
from pipeline.ollama_llm import OllamaChatModel
import chromadb
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize page configuration
st.set_page_config(
    page_title="Lutheran Confessional Assistant",
    page_icon="⛪",
    layout="centered"
)

# Custom premium CSS injection
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@0,600;0,700;1,400&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0F172A;
        color: #F8FAFC;
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #F59E0B, #EF4444);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1rem;
        font-style: italic;
        color: #94A3B8;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Custom styles for details/summary */
    details {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        margin-top: 1rem;
        transition: border-color 0.3s ease;
    }

    details:hover {
        border-color: #F59E0B;
    }

    summary {
        font-weight: 600;
        font-size: 1.05rem;
        color: #F59E0B;
        cursor: pointer;
        outline: none;
    }

    .status-indicator {
        font-size: 0.9rem;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
    }

    .status-ok {
        background-color: #059669;
        color: white;
    }

    .status-err {
        background-color: #DC2626;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_settings():
    return Settings()

@st.cache_resource
def load_db_engine(_settings):
    try:
        engine = get_engine(_settings)
        if check_connection(engine):
            return engine
        return None
    except Exception as e:
        logger.error("Failed to connect to database: %s", e)
        return None

@st.cache_resource
def load_chroma_client(_settings):
    try:
        return chromadb.PersistentClient(path=_settings.chroma_db_path)
    except Exception as e:
        logger.error("Failed to connect to ChromaDB: %s", e)
        return None

@st.cache_resource
def load_embedding_model():
    try:
        # Pass embed_model=None to let VectorIndexer create the model, or instantiate it directly
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        logger.error("Failed to load embedding model: %s", e)
        return None

# Load resources
settings = load_settings()
db_engine = load_db_engine(settings)
chroma_client = load_chroma_client(settings)
embed_model = load_embedding_model()

from pipeline.ollama_llm import start_ollama_server, ensure_model_loaded

# Trigger automatic server start on load
start_ollama_server(settings.ollama_base_url)

def check_ollama_running(url):
    import requests
    try:
        res = requests.get(url, timeout=0.5)
        return res.status_code == 200
    except Exception:
        return False

# Header Section
st.markdown('<div class="main-title">Lutheran Confessional Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Faithful theological guidance grounded in Scripture and the Book of Concord</div>', unsafe_allow_html=True)

# Sidebar Configuration & Status
st.sidebar.title("Settings & Status")

# Decoupling Bible Translation selector
selected_version = st.sidebar.selectbox(
    "Primary Bible Translation",
    options=["WEB", "KJV", "MKJV"],
    index=["WEB", "KJV", "MKJV"].index(settings.primary_search_version)
)
if selected_version != settings.primary_search_version:
    settings.primary_search_version = selected_version

# System status indicators
db_ok = db_engine is not None
chroma_ok = chroma_client is not None
embed_ok = embed_model is not None
ollama_ok = check_ollama_running(settings.ollama_base_url)

# Model verification check
model_ready = False
model_msg = ""
if ollama_ok:
    try:
        import requests
        res = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=1)
        if res.status_code == 200:
            models = res.json().get("models", [])
            model_names = [m.get("name") for m in models]
            model_base_names = [name.split(":")[0] for name in model_names if name]
            if settings.ollama_model in model_names or settings.ollama_model in model_base_names:
                model_ready = True
    except Exception as e:
        model_msg = str(e)

st.sidebar.subheader("System Status")
st.sidebar.markdown(f"Database: <span class='status-indicator {'status-ok' if db_ok else 'status-err'}'>{'Connected' if db_ok else 'Disconnected'}</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"ChromaDB: <span class='status-indicator {'status-ok' if chroma_ok else 'status-err'}'>{'Connected' if chroma_ok else 'Disconnected'}</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"Embedding Model: <span class='status-indicator {'status-ok' if embed_ok else 'status-err'}'>{'Loaded' if embed_ok else 'Failed'}</span>", unsafe_allow_html=True)

if ollama_ok:
    st.sidebar.markdown(f"Ollama LLM: <span class='status-indicator status-ok'>Connected</span>", unsafe_allow_html=True)
    if not model_ready:
        st.sidebar.warning(f"⚠️ Model '{settings.ollama_model}' is not installed locally.")
        if st.sidebar.button("📥 Download Llama 3 Model (4.7 GB)"):
            with st.spinner("Downloading Llama 3 (this may take several minutes)..."):
                success, msg = ensure_model_loaded(settings.ollama_base_url, settings.ollama_model)
                if success:
                    st.sidebar.success("Model downloaded successfully!")
                    st.rerun()
                else:
                    st.sidebar.error(f"Download failed: {msg}")
else:
    st.sidebar.markdown(f"Ollama LLM: <span class='status-indicator status-err'>Simulated Mode</span>", unsafe_allow_html=True)
    st.sidebar.warning("⚠️ Local Ollama is offline. The assistant is running in Simulated Mode (RAG context retrieval is fully active, but responses are simulated).")
    st.sidebar.markdown(
        """
        **To run local inference:**
        1. Download [Ollama](https://ollama.com).
        2. Run the model in your terminal:
           ```bash
           ollama run llama3
           ```
        """
    )

st.sidebar.subheader("Configuration")
st.sidebar.write(f"**Ollama Model:** `{settings.ollama_model}`")
st.sidebar.write(f"**Ollama Base URL:** `{settings.ollama_base_url}`")
st.sidebar.write(f"**Chroma DB Path:** `{settings.chroma_db_path}`")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Handle user input
query = st.chat_input("Ask a question about Lutheran doctrine:")
if query:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate assistant response
    with st.chat_message("assistant"):
        if not (db_ok and chroma_ok and embed_ok):
            error_msg = (
                "⚠️ System connection issues detected. Please make sure PostgreSQL, ChromaDB, "
                "and the embedding model are properly running."
            )
            st.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            with st.spinner("Searching scriptural and confessional context..."):
                try:
                    # Choose LLM client based on status
                    if ollama_ok:
                        llm = OllamaChatModel(model_name=settings.ollama_model, base_url=settings.ollama_base_url)
                    else:
                        from pipeline.ollama_llm import SimulatedChatModel
                        llm = SimulatedChatModel()
                    
                    response = run_orchestrator(
                        chroma_client=chroma_client,
                        db_engine=db_engine,
                        llm=llm,
                        query=query,
                        embed_model=embed_model
                    )
                    
                    st.markdown(response, unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    err_text = f"❌ Error executing RAG pipeline: {e}"
                    st.markdown(err_text)
                    st.session_state.messages.append({"role": "assistant", "content": err_text})
