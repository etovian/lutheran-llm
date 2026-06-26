import logging
from database.queries import fetch_parallel_verses_and_lexicon

logger = logging.getLogger(__name__)

def retrieve_context(chroma_client, db_engine, query: str, embed_model, db_lookup_func=None) -> dict:
    """
    Retrieve semantic context from ChromaDB collections and fetch parallel bible verses
    and lexicon definitions from the relational database.
    
    Args:
        chroma_client: The ChromaDB client instance.
        db_engine: SQLAlchemy database engine instance.
        query (str): The search query text.
        embed_model: Embedding model instance to generate query vectors.
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
        conf_res = conf_collection.query(query_embeddings=[query_embedding], n_results=1)
        
        conf_docs = conf_res.get("documents", [[]])[0] if conf_res.get("documents") else []
        conf_metas = conf_res.get("metadatas", [[]])[0] if conf_res.get("metadatas") else []
        
        confessional_chunks = []
        for doc, meta in zip(conf_docs, conf_metas):
            confessional_chunks.append({
                "text": doc,
                "citation": meta.get("citation", "Unknown Citation")
            })
            
        bib_collection = chroma_client.get_collection("biblical_collection")
        bib_res = bib_collection.query(query_embeddings=[query_embedding], n_results=1)
        
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
