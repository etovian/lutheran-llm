import logging
import html
from typing import Any, Callable, Dict, List, Optional
from config.settings import Settings
from database.queries import fetch_parallel_verses_and_lexicon
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
    db_lookup_func: Optional[Callable[..., Any]] = None
) -> Dict[str, Any]:
    """
    Retrieve semantic context from ChromaDB collections and fetch parallel bible verses
    and lexicon definitions from the relational database.
    
    Args:
        chroma_client: The ChromaDB client instance.
        db_engine: SQLAlchemy database engine instance.
        query (str): The search query text.
        embed_model: Embedding model instance to generate query vectors.
        confessional_k (int): Number of confessional documents to retrieve.
        biblical_k (int): Number of biblical documents to retrieve.
        db_lookup_func: Function to lookup relational database records, defaults to 
                        fetch_parallel_verses_and_lexicon.
                        
    Returns:
        dict: A dictionary containing 'confessional' chunk details and parallel 'scriptures' info.
    """
    if db_lookup_func is None:
        db_lookup_func = fetch_parallel_verses_and_lexicon
        
    if confessional_k is None:
        confessional_k = settings.rag_confessional_k
    if biblical_k is None:
        biblical_k = settings.rag_biblical_k
        
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
        bib_res = bib_collection.query(query_embeddings=[query_embedding], n_results=biblical_k)
        
        bib_metas = bib_res.get("metadatas", [[]])[0] if bib_res.get("metadatas") else []
        
        scriptures = []
        for meta in bib_metas:
            verse_id = meta.get("verse_id")
            if verse_id is not None:
                scripture_data = db_lookup_func(db_engine, verse_id)
                if scripture_data:
                    scripture_data = dict(scripture_data)
                    book_name = meta.get("book_name", "Unknown Book")
                    chapter = meta.get("chapter", 0)
                    verse_number = meta.get("verse_number", 0)
                    scripture_data["citation"] = f"{book_name} {chapter}:{verse_number}"
                    scripture_data["book_name"] = book_name
                    scripture_data["chapter"] = chapter
                    scripture_data["verse_number"] = verse_number
                    scripture_data["address_code"] = meta.get("address_code")
                    scriptures.append(scripture_data)
                    
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
        
    # Scripture Context (Translations only, omitting lexicon to optimize CPU latency)
    lines.append("--- SCRIPTURE CONTEXT ---")
    scriptures = retrieved_ctx.get("scriptures", [])
    if scriptures:
        for scripture in scriptures:
            citation = scripture.get("citation", "Unknown Scripture")
            lines.append(f"Citation: {citation}")
            translations = scripture.get("translations", {})
            if translations:
                for ver, text_val in translations.items():
                    lines.append(f"[{ver}]: {text_val}")
            else:
                lines.append("No parallel scripture translations found.")
            lines.append("")
    else:
        lines.append("No parallel scripture translations found.")
        
    return "\n".join(lines)


def format_deep_dive_details(retrieved_ctx: Dict[str, Any]) -> str:
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
            lines.append(f"<p><strong>{citation}</strong>: <em>\"{text_val}\"</em></p>")
    else:
        lines.append("<p>No confessional citations found.</p>")
        
    # Parallel Bible Translations
    lines.append("<h4>Parallel Bible Translations</h4>")
    scriptures = retrieved_ctx.get("scriptures", [])
    if scriptures:
        for scripture in scriptures:
            citation = html.escape(scripture.get("citation", "Unknown Scripture"))
            lines.append(f"<h5>{citation}</h5>")
            translations = scripture.get("translations", {})
            if translations:
                lines.append("<ul>")
                for ver, text_val in translations.items():
                    escaped_ver = html.escape(ver)
                    escaped_text = html.escape(text_val)
                    lines.append(f"<li>[{escaped_ver}]: {escaped_text}</li>")
                lines.append("</ul>")
            else:
                lines.append("<p>No parallel scripture translations found.</p>")
    else:
        lines.append("<p>No parallel scripture translations found.</p>")
        
    # Original Language Word Analysis
    lines.append("<h4>Original Language Word Analysis</h4>")
    if scriptures:
        has_any_lexicon = False
        for scripture in scriptures:
            lexicon = scripture.get("lexicon", [])
            if lexicon:
                has_any_lexicon = True
                citation = html.escape(scripture.get("citation", "Unknown Scripture"))
                lines.append(f"<h5>{citation}</h5>")
                lines.append("<ul>")
                for lex in lexicon:
                    word_text = html.escape(lex.get('word_text', ''))
                    lemma = html.escape(lex.get('lemma', ''))
                    strongs_number = html.escape(lex.get('strongs_number', ''))
                    definition = html.escape(lex.get('definition', ''))
                    lines.append(
                        f"<li>Word: {word_text}, Lemma: {lemma}, "
                        f"Strongs: {strongs_number}, "
                        f"Definition: {definition}</li>"
                    )
                lines.append("</ul>")
        if not has_any_lexicon:
            lines.append("<p>No lexicon analysis found.</p>")
    else:
        lines.append("<p>No lexicon analysis found.</p>")
        
    lines.append("</details>")
    return "\n".join(lines)


def run_orchestrator(
    chroma_client: Any, 
    db_engine: Any, 
    llm: Any, 
    query: str, 
    embed_model: Any,
    confessional_k: Optional[int] = None,
    biblical_k: Optional[int] = None
) -> str:
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
        
    Returns:
        str: Synthesized response from the LLM.
    """
    if detect_pastoral_crisis(query):
        logger.info("Pastoral crisis detected for query: %s. Preempting loop.", query)
        return get_redirection_response()

    try:
        retrieved_ctx = retrieve_context(
            chroma_client, 
            db_engine, 
            query, 
            embed_model,
            confessional_k=confessional_k,
            biblical_k=biblical_k
        )
        formatted_ctx = format_llm_context(retrieved_ctx)
        
        system_prompt = SYSTEM_PROMPT.format(context=formatted_ctx)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
        
        response = llm.invoke(messages)
        response_content = response.content if hasattr(response, "content") else str(response)
        
        # Post-process response to ensure robust HTML collapsible structure
        if "<details>" in response_content and "</details>" in response_content:
            return response_content
            
        if "<details>" in response_content:
            response_content = response_content.split("<details>")[0].strip()
            
        # Programmatically construct and append deep-dive details
        details_html = format_deep_dive_details(retrieved_ctx)
        return f"{response_content}\n\n{details_html}"
        
    except Exception as e:
        logger.error("Failed to run orchestrator execution loop: %s", e, exc_info=True)
        raise

