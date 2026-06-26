import logging
from database.queries import fetch_parallel_verses_and_lexicon
from langchain_core.messages import SystemMessage, HumanMessage
from pipeline.prompt import SYSTEM_PROMPT
from pipeline.guardrails import detect_pastoral_crisis, get_redirection_response

logger = logging.getLogger(__name__)

def retrieve_context(
    chroma_client, 
    db_engine, 
    query: str, 
    embed_model, 
    confessional_k: int = 1, 
    biblical_k: int = 1, 
    db_lookup_func=None
) -> dict:
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
        dict: A dictionary containing 'confessional' chunk details and parallel 'scripture' info.
    """
    if db_lookup_func is None:
        db_lookup_func = fetch_parallel_verses_and_lexicon
        
    try:
        query_embedding = embed_model.encode(query).tolist()
        
        conf_collection = chroma_client.get_collection("confessional_collection")
        conf_res = conf_collection.query(query_embeddings=[query_embedding], n_results=confessional_k)
        
        conf_docs = conf_res.get("documents", [[]])[0] if conf_res.get("documents") else []
        conf_metas = conf_res.get("metadatas", [[]])[0] if conf_res.get("metadatas") else []
        
        confessional_chunks = []
        for doc, meta in zip(conf_docs, conf_metas):
            confessional_chunks.append({
                "text": doc,
                "citation": meta.get("citation", "Unknown Citation")
            })
            
        bib_collection = chroma_client.get_collection("biblical_collection")
        bib_res = bib_collection.query(query_embeddings=[query_embedding], n_results=biblical_k)
        
        bib_metas = bib_res.get("metadatas", [[]])[0] if bib_res.get("metadatas") else []
        
        scripture_ctx = {}
        if bib_metas and len(bib_metas) > 0:
            verse_id = bib_metas[0].get("verse_id")
            if verse_id is not None:
                scripture_ctx = db_lookup_func(db_engine, verse_id)
                
    except Exception as e:
        logger.error("Failed to retrieve context for query %r: %s", query, e, exc_info=True)
        raise
        
    return {
        "confessional": confessional_chunks,
        "scripture": scripture_ctx
    }


def format_context(retrieved_ctx: dict) -> str:
    """
    Format retrieved confessional chunks and scripture context into a structured string.
    
    Args:
        retrieved_ctx (dict): The dictionary containing retrieved context.
        
    Returns:
        str: Formatted context string.
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
        
    # Scripture Context
    lines.append("--- SCRIPTURE CONTEXT ---")
    scripture = retrieved_ctx.get("scripture", {})
    translations = scripture.get("translations", {})
    if translations:
        for ver, text_val in translations.items():
            lines.append(f"[{ver}]: {text_val}")
        lines.append("")
    else:
        lines.append("No parallel scripture translations found.")
        
    # Lexicon
    lexicon = scripture.get("lexicon", [])
    if lexicon:
        lines.append("Original language word analysis:")
        for lex in lexicon:
            lines.append(
                f"- Word: {lex.get('word_text')}, Lemma: {lex.get('lemma')}, "
                f"Strongs: {lex.get('strongs_number')}, "
                f"Definition: {lex.get('definition')}"
            )
    else:
        lines.append("No lexicon analysis found.")
        
    return "\n".join(lines)


def format_llm_context(retrieved_ctx: dict) -> str:
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
    scripture = retrieved_ctx.get("scripture", {})
    translations = scripture.get("translations", {})
    if translations:
        for ver, text_val in translations.items():
            lines.append(f"[{ver}]: {text_val}")
        lines.append("")
    else:
        lines.append("No parallel scripture translations found.")
        
    return "\n".join(lines)


def format_deep_dive_details(retrieved_ctx: dict) -> str:
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
            citation = chunk.get("citation", "Book of Concord")
            text_val = chunk.get("text", "")
            lines.append(f"<p><strong>{citation}</strong>: <em>\"{text_val}\"</em></p>")
    else:
        lines.append("<p>No confessional citations found.</p>")
        
    # Parallel Bible Translations
    lines.append("<h4>Parallel Bible Translations</h4>")
    scripture = retrieved_ctx.get("scripture", {})
    translations = scripture.get("translations", {})
    if translations:
        lines.append("<ul>")
        for ver, text_val in translations.items():
            lines.append(f"<li>[{ver}]: {text_val}</li>")
        lines.append("</ul>")
    else:
        lines.append("<p>No parallel scripture translations found.</p>")
        
    # Original Language Word Analysis
    lines.append("<h4>Original Language Word Analysis</h4>")
    lexicon = scripture.get("lexicon", [])
    if lexicon:
        lines.append("<ul>")
        for lex in lexicon:
            lines.append(
                f"<li>Word: {lex.get('word_text')}, Lemma: {lex.get('lemma')}, "
                f"Strongs: {lex.get('strongs_number')}, "
                f"Definition: {lex.get('definition')}</li>"
            )
        lines.append("</ul>")
    else:
        lines.append("<p>No lexicon analysis found.</p>")
        
    lines.append("</details>")
    return "\n".join(lines)


def run_orchestrator(chroma_client, db_engine, llm, query: str, embed_model) -> str:
    """
    Execute the RAG retrieval, context formatting, and LLM orchestration loop.
    
    Args:
        chroma_client: Chroma client instance.
        db_engine: SQLAlchemy database engine instance.
        llm: Language model client instance.
        query (str): The user query.
        embed_model: Embedding model instance.
        
    Returns:
        str: Synthesized response from the LLM.
    """
    if detect_pastoral_crisis(query):
        logger.info("Pastoral crisis detected for query: %s. Preempting loop.", query)
        return get_redirection_response()

    try:
        retrieved_ctx = retrieve_context(chroma_client, db_engine, query, embed_model)
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

