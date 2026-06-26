from sentence_transformers import SentenceTransformer

class VectorIndexer:
    """
    VectorIndexer manages embedding generation and storage of documents 
    in ChromaDB collections (confessional and biblical scripture).
    """
    def __init__(self, chroma_client, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the indexer with a Chroma client and embedding model.
        
        Args:
            chroma_client: The ChromaDB client instance.
            model_name (str): Name of the SentenceTransformers embedding model to use.
        """
        self.chroma_client = chroma_client
        self.embed_model = SentenceTransformer(model_name)
        
    def index_confessional(self, chunk: dict):
        """
        Generate embedding for a confessional chunk and add it to the confessional collection.
        
        Args:
            chunk (dict): A dictionary containing 'text', 'book', 'article_id', 'paragraph_number', and 'citation'.
        """
        collection = self.chroma_client.get_collection("confessional_collection")
        embedding = self.embed_model.encode(chunk["text"]).tolist()
        
        collection.add(
            documents=[chunk["text"]],
            embeddings=[embedding],
            metadatas=[{
                "book": chunk["book"],
                "article_id": chunk["article_id"],
                "paragraph_number": chunk["paragraph_number"],
                "citation": chunk["citation"]
            }],
            ids=[f"{chunk['book']}_{chunk['article_id']}_{chunk['paragraph_number']}"]
        )

    def index_biblical(self, verse: dict):
        """
        Generate embedding for a biblical verse and add it to the biblical collection.
        
        Args:
            verse (dict): A dictionary containing 'text', 'verse_id', 'address_code', 'book_name', 'chapter', and 'verse_number'.
        """
        collection = self.chroma_client.get_collection("biblical_collection")
        embedding = self.embed_model.encode(verse["text"]).tolist()
        
        collection.add(
            documents=[verse["text"]],
            embeddings=[embedding],
            metadatas=[{
                "verse_id": verse["verse_id"],
                "address_code": verse["address_code"],
                "book_name": verse["book_name"],
                "chapter": verse["chapter"],
                "verse_number": verse["verse_number"]
            }],
            ids=[str(verse["verse_id"])]
        )
