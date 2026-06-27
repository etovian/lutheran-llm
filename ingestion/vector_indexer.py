import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class VectorIndexer:
    """
    VectorIndexer manages embedding generation and storage of documents 
    in ChromaDB collections (confessional and scripture).
    """
    def __init__(self, chroma_client, model_name: str = "all-MiniLM-L6-v2", embed_model=None):
        """
        Initialize the indexer with a Chroma client and embedding model.
        
        Args:
            chroma_client: The ChromaDB client instance.
            model_name (str): Name of the SentenceTransformers embedding model to use.
            embed_model: Pre-initialized embedding model (useful for testing/injection).
        """
        self.chroma_client = chroma_client
        if embed_model is not None:
            self.embed_model = embed_model
        else:
            self.embed_model = SentenceTransformer(model_name)
            
    def index_confessional_batch(self, chunks: list[dict]):
        """
        Index a list of confessional chunks in a single batch.
        
        Args:
            chunks (list[dict]): List of dictionaries containing confessional metadata and text.
        """
        if not chunks:
            return
        try:
            collection = self.chroma_client.get_collection("confessional_collection")
            texts = [c["text"] for c in chunks]
            embeddings = self.embed_model.encode(texts).tolist()
            metadatas = [{
                "book": c["book"],
                "article_id": c["article_id"],
                "paragraph_number": c["paragraph_number"],
                "citation": c["citation"]
            } for c in chunks]
            seen = {}
            ids = []
            for c in chunks:
                base_id = f"{c['book']}_{c['article_id']}_{c['paragraph_number']}"
                if base_id in seen:
                    seen[base_id] += 1
                    ids.append(f"{base_id}_{seen[base_id]}")
                else:
                    seen[base_id] = 0
                    ids.append(base_id)
            
            collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            logger.error("Failed to index confessional batch: %s", e, exc_info=True)
            raise

    def index_confessional(self, chunk: dict):
        """
        Index a single confessional chunk.
        
        Args:
            chunk (dict): Dictionary containing 'text', 'book', 'article_id', 'paragraph_number', and 'citation'.
        """
        self.index_confessional_batch([chunk])

    def index_biblical_batch(self, verses: list[dict]):
        """
        Index a list of biblical verses in a single batch.
        
        Args:
            verses (list[dict]): List of dictionaries containing biblical metadata and text.
        """
        if not verses:
            return
        try:
            collection = self.chroma_client.get_collection("biblical_collection")
            texts = [v["text"] for v in verses]
            embeddings = self.embed_model.encode(texts).tolist()
            metadatas = [{
                "verse_id": v["verse_id"],
                "address_code": v["address_code"],
                "book_name": v["book_name"],
                "chapter": v["chapter"],
                "verse_number": v["verse_number"]
            } for v in verses]
            ids = [str(v["verse_id"]) for v in verses]
            
            collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            logger.error("Failed to index biblical batch: %s", e, exc_info=True)
            raise

    def index_biblical(self, verse: dict):
        """
        Index a single biblical verse.
        
        Args:
            verse (dict): Dictionary containing 'text', 'verse_id', 'address_code', 'book_name', 'chapter', and 'verse_number'.
        """
        self.index_biblical_batch([verse])
