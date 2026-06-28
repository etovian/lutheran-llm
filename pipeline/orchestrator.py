import logging
import html
from typing import Any, Callable, Dict, List, Optional
from config.settings import Settings

class OrchestratorResponse(str):
    """A custom string subclass that holds extra attributes from orchestrator run."""
    def __new__(cls, content, summary=None, retrieved_ctx=None):
        obj = super().__new__(cls, content)
        obj.summary = summary if summary is not None else ""
        obj.retrieved_ctx = retrieved_ctx if retrieved_ctx is not None else {}
        return obj
from database.queries import fetch_single_translation
from langchain_core.messages import SystemMessage, HumanMessage
from pipeline.prompt import SYSTEM_PROMPT
from pipeline.guardrails import detect_pastoral_crisis, get_redirection_response

logger = logging.getLogger(__name__)
settings = Settings()

def retrieve_context(
    chroma_client: Any,
    db_engine: Any,
    query: str,
    embed_model: Any,
    confessional_k: Optional[int] = None,
    biblical_k: Optional[int] = None,
    db_lookup_func: Optional[Callable[..., Any]] = None,
    primary_translation: str = "WEB"
) -> Dict[str, Any]:
    """
    Retrieve semantic context from ChromaDB collections and fetch a single Bible translation
    per verse from the relational database.

    Args:
        chroma_client: The ChromaDB client instance.
        db_engine: SQLAlchemy database engine instance.
        query (str): The search query text.
        embed_model: Embedding model instance to generate query vectors.
        confessional_k (int): Number of confessional documents to retrieve.
        biblical_k (int): Number of biblical documents to retrieve.
        db_lookup_func: Optional callable for test injection; receives (db_engine, verse_id)
                        and returns a string (the verse text). In production, fetch_single_translation
                        is called directly.
        primary_translation (str): Which Bible translation to cache (e.g. "WEB", "KJV").

    Returns:
        dict: A dictionary containing 'confessional' chunk details and 'scriptures' with
              verse_id, primary_translation, and cached_text.
    """
    if confessional_k is None:
        confessional_k = settings.rag_confessional_k
    bib_limit = biblical_k if biblical_k is not None else settings.rag_biblical_max_pool
        
    try:
        query_embedding = embed_model.encode(query).tolist()
        
        conf_coll_name = getattr(settings, "chroma_confessional_collection", "confessional_collection")
        conf_collection = chroma_client.get_collection(conf_coll_name)
        conf_res = conf_collection.query(query_embeddings=[query_embedding], n_results=confessional_k)
        
        conf_docs = conf_res.get("documents", [[]])[0] if conf_res.get("documents") else []
        conf_metas = conf_res.get("metadatas", [[]])[0] if conf_res.get("metadatas") else []
        
        confessional_chunks = []
        for doc, meta in zip(conf_docs, conf_metas):
            confessional_chunks.append({
                "text": doc,
                "citation": meta.get("citation", "Unknown Citation")
            })
            
        bib_coll_name = getattr(settings, "chroma_biblical_collection", "biblical_collection")
        bib_collection = chroma_client.get_collection(bib_coll_name)
        bib_res = bib_collection.query(query_embeddings=[query_embedding], n_results=bib_limit)
        
        bib_metas = bib_res.get("metadatas", [[]])[0] if bib_res.get("metadatas") else []
        bib_distances = bib_res.get("distances", [[]])[0] if bib_res.get("distances") else [0.0] * len(bib_metas)
        
        scriptures = []
        for meta, dist in zip(bib_metas, bib_distances):
            if dist > settings.rag_biblical_distance_threshold:
                continue
            verse_id = meta.get("verse_id")
            if verse_id is not None:
                if db_lookup_func is not None:
                    # Test injection path: db_lookup_func returns a string directly
                    cached_text = db_lookup_func(db_engine, verse_id)
                else:
                    cached_text = fetch_single_translation(db_engine, verse_id, primary_translation)
                book_name = meta.get("book_name", "Unknown Book")
                chapter = meta.get("chapter", 0)
                verse_number = meta.get("verse_number", 0)
                scriptures.append({
                    "citation": f"{book_name} {chapter}:{verse_number}",
                    "verse_id": int(verse_id),
                    "primary_translation": primary_translation,
                    "cached_text": cached_text,
                    "book_name": book_name,
                    "chapter": chapter,
                    "verse_number": verse_number,
                    "address_code": meta.get("address_code"),
                    "distance": dist
                })
        logger.info(
            "Retrieved %d biblical passages; %d passed distance threshold <= %f",
            len(bib_distances), len(scriptures), settings.rag_biblical_distance_threshold
        )
                    
    except Exception as e:
        logger.error("Failed to retrieve context for query %r: %s", query, e, exc_info=True)
        raise
        
    return {
        "confessional": confessional_chunks,
        "scriptures": scriptures
    }


def format_llm_context(retrieved_ctx: Dict[str, Any]) -> str:
    """
    Format retrieved confessional chunks and scripture context (excluding lexicon)
    into a structured string to minimize context window size for LLM generation.
    """
    lines = []
    
    # Confessional chunks
    lines.append("--- CONFESSIONAL CONTEXT ---")
    confessional_chunks = retrieved_ctx.get("confessional", [])
    if confessional_chunks:
        for chunk in confessional_chunks:
            lines.append(f"Source: {chunk.get('citation')}")
            lines.append(f"Content: {chunk.get('text')}\n")
    else:
        lines.append("No confessional context found.\n")
        
    # Scripture Context (Single translation to minimize context window)
    lines.append("--- SCRIPTURE CONTEXT ---")
    scriptures = retrieved_ctx.get("scriptures", [])
    if scriptures:
        for scripture in scriptures:
            citation = scripture.get("citation", "Unknown Scripture")
            ver = scripture.get("primary_translation", "WEB")
            text_val = scripture.get("cached_text", "")
            lines.append(f"Citation: {citation}")
            lines.append(f"[{ver}]: {text_val}")
            lines.append("")
    else:
        lines.append("No scripture context found.")
        
    return "\n".join(lines)


def format_deep_dive_details(
    retrieved_ctx: Dict[str, Any],
    primary_translation: str = "WEB",
    db_engine: Any = None
) -> str:
    """
    Programmatically construct the HTML collapsible deep-dive details block.
    """
    lines = []
    lines.append("<details>")
    lines.append("<summary>Theological Depth</summary>")
    
    # Book of Concord Citations
    lines.append("<h4>Book of Concord Citations</h4>")
    confessional = retrieved_ctx.get("confessional", [])
    if confessional:
        for chunk in confessional:
            citation = html.escape(chunk.get("citation", "Book of Concord"))
            text_val = html.escape(chunk.get("text", ""))
            lines.append(
                f'<details class="boc-detail" style="margin-bottom: 0.6rem; margin-left: 0.5rem; border-left: 2px solid #F59E0B; padding: 0.3rem 0.6rem;">'
                f'  <summary style="font-weight: 500; font-size: 0.95rem; color: #E2E8F0; cursor: pointer;">{citation}</summary>'
                f'  <p style="margin-top: 0.4rem; font-style: italic; color: #94A3B8; font-size: 0.9rem;">"{text_val}"</p>'
                f'</details>'
            )
    else:
        lines.append("<p>No confessional citations found.</p>")
        
    # Bible Passages
    lines.append("<h4>Bible Passages</h4>")
    scriptures = retrieved_ctx.get("scriptures", [])
    if scriptures:
        for scripture in scriptures:
            citation = html.escape(scripture.get("citation", "Unknown Scripture"))
            verse_id = scripture.get("verse_id")
            cached_ver = scripture.get("primary_translation", "WEB")
            cached_text = scripture.get("cached_text", "")
            
            if primary_translation == cached_ver or db_engine is None:
                # Use cached text (same version, or no DB available/tests)
                verse_text = cached_text
            else:
                # On-demand DB lookup for the alternate translation
                verse_text = fetch_single_translation(db_engine, verse_id, primary_translation)
                if not verse_text:
                    verse_text = cached_text  # graceful fallback
            
            escaped_translation = html.escape(primary_translation)
            escaped_text = html.escape(verse_text)
            lines.append(f"<p><strong>{citation} ({escaped_translation})</strong>: {escaped_text}</p>")
    else:
        lines.append("<p>No parallel scripture translations found.</p>")
        
    lines.append("</details>")
    return "\n".join(lines)


def run_orchestrator(
    chroma_client: Any, 
    db_engine: Any, 
    llm: Any, 
    query: str, 
    embed_model: Any,
    confessional_k: Optional[int] = None,
    biblical_k: Optional[int] = None,
    primary_translation: str = "WEB"
) -> OrchestratorResponse:
    """
    Execute the RAG retrieval, context formatting, and LLM orchestration loop.
    
    Args:
        chroma_client: Chroma client instance.
        db_engine: SQLAlchemy database engine instance.
        llm: Language model client instance.
        query (str): The user query.
        embed_model: Embedding model instance.
        confessional_k (int): Number of confessional documents to retrieve.
        biblical_k (int): Number of biblical documents to retrieve.
        primary_translation (str): Primary Bible translation version.
        
    Returns:
        OrchestratorResponse: Synthesized response from the LLM with summary and retrieved_ctx metadata.
    """
    if detect_pastoral_crisis(query):
        logger.info("Pastoral crisis detected for query: %s. Preempting loop.", query)
        redirection = get_redirection_response()
        return OrchestratorResponse(redirection, summary=redirection, retrieved_ctx={"confessional": [], "scriptures": []})

    try:
        retrieved_ctx = retrieve_context(
            chroma_client, 
            db_engine, 
            query, 
            embed_model,
            confessional_k=confessional_k,
            biblical_k=biblical_k,
            primary_translation=primary_translation
        )
        formatted_ctx = format_llm_context(retrieved_ctx)
        
        system_prompt = SYSTEM_PROMPT.format(context=formatted_ctx)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
        
        # Invoke the LLM
        response = llm.invoke(messages)
        response_content = response.content if hasattr(response, "content") else str(response)
        
        summary = response_content
        if "<details>" in response_content:
            summary = response_content.split("<details>")[0].strip()
            
        # Programmatically construct and append deep-dive details
        details_html = format_deep_dive_details(retrieved_ctx, primary_translation, db_engine=db_engine)
        full_res = f"{summary}\n\n{details_html}"
        return OrchestratorResponse(full_res, summary=summary, retrieved_ctx=retrieved_ctx)
        
    except Exception as e:
        logger.error("Failed to run orchestrator execution loop: %s", e, exc_info=True)
        raise
