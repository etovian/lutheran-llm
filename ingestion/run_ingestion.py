import logging
from config.settings import Settings
from database.connection import get_engine
from ingestion.vector_indexer import VectorIndexer
from ingestion.parse_bible import parse_original_word_tokens
from sqlalchemy import text
import chromadb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Core Seed Data
CONFESSlONAL_SEEDS = [
    {
        "text": (
            "Also they teach that men cannot be justified before God by their own strength, merits, or works, "
            "but are freely justified for Christ's sake, through faith, when they believe that they are received "
            "into favor, and that their sins are forgiven for Christ's sake, who, by His death, has made satisfaction "
            "for our sins. This faith God imputes for righteousness in His sight. Rom. 3 and 4."
        ),
        "book": "Augsburg Confession",
        "article_id": "AC_IV",
        "paragraph_number": 1,
        "citation": "Augsburg Confession, Article IV, Paragraph 1"
    },
    {
        "text": (
            "Also they teach that one holy Church is to continue forever. The Church is the congregation of saints, "
            "in which the Gospel is rightly taught and the Sacraments are rightly administered."
        ),
        "book": "Augsburg Confession",
        "article_id": "AC_VII",
        "paragraph_number": 1,
        "citation": "Augsburg Confession, Article VII, Paragraph 1"
    },
    {
        "text": (
            "Our teachers are falsely accused of forbidding good works. For their published writings on the Ten "
            "Commandments, and others of like import, bear witness that they have taught to good purpose concerning "
            "all estates and duties of life, as to what estates and graves of life, and what works in any calling, "
            "be pleasing to God."
        ),
        "book": "Augsburg Confession",
        "article_id": "AC_XX",
        "paragraph_number": 1,
        "citation": "Augsburg Confession, Article XX, Paragraph 1"
    }
]

BIBLE_BOOKS = [
    {"book_id": 45, "name": "Romans", "testament": "NT"},
    {"book_id": 49, "name": "Ephesians", "testament": "NT"}
]

STRONGS_CONCORDANCE = [
    {"strongs_number": "G3049", "pronunciation": "log-id'-zom-a-hee", "definition": "to reckon, count, compute, calculate, take into account", "derivation": "from λογος"},
    {"strongs_number": "G3767", "pronunciation": "oon", "definition": "therefore, then, now", "derivation": "a particle of transition"},
    {"strongs_number": "G4102", "pronunciation": "pis'-tis", "definition": "faith, belief, trust, confidence", "derivation": "from πειθω"},
    {"strongs_number": "G1344", "pronunciation": "dik-ah-yo'-o", "definition": "to justify, declare righteous, show to be righteous", "derivation": "from δικαιος"},
    {"strongs_number": "G444", "pronunciation": "anth'-ro-pos", "definition": "man, human being, person", "derivation": "from ανηρ"},
    {"strongs_number": "G5565", "pronunciation": "kho-rece'", "definition": "apart from, separate, without", "derivation": "from χωρος"},
    {"strongs_number": "G2041", "pronunciation": "er'-gon", "definition": "work, deed, action", "derivation": "from εργω"},
    {"strongs_number": "G3551", "pronunciation": "nom'-os", "definition": "law, rule, principle", "derivation": "from νεμω"},
    {"strongs_number": "G3588", "pronunciation": "tay", "definition": "the, this, that", "derivation": "article"},
    {"strongs_number": "G1063", "pronunciation": "gar", "definition": "for, indeed, because", "derivation": "conjunction"},
    {"strongs_number": "G5485", "pronunciation": "khar'-ece", "definition": "grace, favor, loving-kindness", "derivation": "from χαιρω"},
    {"strongs_number": "G2075", "pronunciation": "es-te'", "definition": "ye are", "derivation": "from ειμι"},
    {"strongs_number": "G4982", "pronunciation": "so'-zo", "definition": "to save, deliver, make whole, preserve", "derivation": "from σως"},
    {"strongs_number": "G1223", "pronunciation": "dee-ah'", "definition": "through, by means of, on account of", "derivation": "preposition"},
    {"strongs_number": "G3756", "pronunciation": "ook", "definition": "not, no", "derivation": "negative particle"},
    {"strongs_number": "G1537", "pronunciation": "ex", "definition": "out of, from, by", "derivation": "preposition"},
    {"strongs_number": "G2443", "pronunciation": "hin'-ah", "definition": "that, in order that, so that", "derivation": "conjunction"},
    {"strongs_number": "G3361", "pronunciation": "may", "definition": "not, lest", "derivation": "particle"},
    {"strongs_number": "G5100", "pronunciation": "tis", "definition": "anyone, someone, a certain one", "derivation": "pronoun"},
    {"strongs_number": "G2744", "pronunciation": "kow-khah'-om-ahee", "definition": "to boast, glory, exult", "derivation": "from αυχην"}
]

BIBLE_VERSES = [
    {
        "verse_id": 28001,
        "book_id": 45,
        "chapter": 3,
        "verse_number": 28,
        "original_verse": "λογιζόμεθα[G3049] οὖν[G3767] πίστει[G4102] δικαιοῦσθαι[G1344] ἄνθρωπον[G444] χωρὶς[G5565] ἔργων[G2041] νόμου[G3551]",
        "address_code": "ROM_3_28",
        "translations": {
            "WEB": "We maintain therefore that a man is justified by faith apart from the works of the law.",
            "KJV": "Therefore we conclude that a man is justified by faith without the deeds of the law.",
            "MKJV": "Therefore we conclude that a man is justified by faith without the works of the law."
        }
    },
    {
        "verse_id": 28145,
        "book_id": 49,
        "chapter": 2,
        "verse_number": 8,
        "original_verse": "τῇ[G3588] γὰρ[G1063] χάριτί[G5485] ἐστε[G2075] σεσωσμένοι[G4982] διὰ[G1223] πίστεως[G4102]",
        "address_code": "EPH_2_8",
        "translations": {
            "WEB": "For by grace you have been saved through faith, and that not of yourselves; it is the gift of God,",
            "KJV": "For by grace are ye saved through faith; and that not of yourselves: it is the gift of God:",
            "MKJV": "For by grace you are saved through faith, and that not of yourselves, it is the gift of God,"
        }
    },
    {
        "verse_id": 28146,
        "book_id": 49,
        "chapter": 2,
        "verse_number": 9,
        "original_verse": "οὐκ[G3756] ἐξ[G1537] ἔργων[G2041] ἵνα[G2443] μή[G3361] τις[G5100] καυχήσηται[G2744]",
        "address_code": "EPH_2_9",
        "translations": {
            "WEB": "not of works, that no one would boast.",
            "KJV": "Not of works, lest any man should boast.",
            "MKJV": "not of works, lest anyone should boast."
        }
    }
]

def run_ingestion():
    settings = Settings()
    engine = get_engine(settings)
    
    # 1. Populate relational database
    logger.info("Connecting to relational database and seeding tables...")
    with engine.connect() as conn:
        # Seed books
        for bk in BIBLE_BOOKS:
            conn.execute(
                text("INSERT INTO book (book_id, name, testament) VALUES (:book_id, :name, :testament) ON CONFLICT DO NOTHING"),
                bk
            )
            
        # Seed strongs concordance definitions
        for st_def in STRONGS_CONCORDANCE:
            conn.execute(
                text(
                    "INSERT INTO strongs_concordance (strongs_number, pronunciation, definition, derivation) "
                    "VALUES (:strongs_number, :pronunciation, :definition, :derivation) ON CONFLICT DO NOTHING"
                ),
                st_def
            )
            
        # Seed verses, translations and parse/seed original language tokens
        for v in BIBLE_VERSES:
            conn.execute(
                text(
                    "INSERT INTO verse (verse_id, book_id, chapter, verse_number, original_verse, address_code) "
                    "VALUES (:verse_id, :book_id, :chapter, :verse_number, :original_verse, :address_code) ON CONFLICT DO NOTHING"
                ),
                {
                    "verse_id": v["verse_id"],
                    "book_id": v["book_id"],
                    "chapter": v["chapter"],
                    "verse_number": v["verse_number"],
                    "original_verse": v["original_verse"],
                    "address_code": v["address_code"]
                }
            )
            
            # Seed translations
            for ver_code, txt_val in v["translations"].items():
                conn.execute(
                    text(
                        "INSERT INTO verse_translation (verse_id, version_code, text) "
                        "VALUES (:verse_id, :version_code, :text) ON CONFLICT DO NOTHING"
                    ),
                    {
                        "verse_id": v["verse_id"],
                        "version_code": ver_code,
                        "text": txt_val
                    }
                )
                
            # Tokenize original language words with Strong's tags and seed original_word table
            word_tokens = parse_original_word_tokens(v["original_verse"], verse_id=v["verse_id"])
            for t in word_tokens:
                conn.execute(
                    text(
                        "INSERT INTO original_word (verse_id, word_index, word_text, lemma, strongs_number) "
                        "VALUES (:verse_id, :word_index, :word_text, :lemma, :strongs_number) "
                        "ON CONFLICT DO NOTHING"
                    ),
                    t
                )
        conn.commit()
    logger.info("Relational database seeding complete.")

    # 2. Populate ChromaDB collections
    logger.info("Initializing ChromaDB collections and generating vector index embeddings...")
    chroma_client = chromadb.PersistentClient(path=settings.chroma_db_path)
    
    # Ensure collections exist
    conf_collection = chroma_client.get_or_create_collection("confessional_collection")
    bib_collection = chroma_client.get_or_create_collection("biblical_collection")
    
    indexer = VectorIndexer(chroma_client=chroma_client)
    
    # Index confessional chunks
    indexer.index_confessional_batch(CONFESSlONAL_SEEDS)
    logger.info(f"Indexed {len(CONFESSlONAL_SEEDS)} Book of Concord chunks in ChromaDB.")
    
    # Index biblical verses (based on user's primary Bible translation preference)
    biblical_indexer_payload = []
    for v in BIBLE_VERSES:
        primary_text = v["translations"].get(settings.primary_search_version, v["translations"]["WEB"])
        biblical_indexer_payload.append({
            "text": primary_text,
            "verse_id": v["verse_id"],
            "address_code": v["address_code"],
            "book_name": "Romans" if v["book_id"] == 45 else "Ephesians",
            "chapter": v["chapter"],
            "verse_number": v["verse_number"]
        })
        
    indexer.index_biblical_batch(biblical_indexer_payload)
    logger.info(f"Indexed {len(biblical_indexer_payload)} scripture verses in ChromaDB.")
    logger.info("Ingestion pipeline successfully completed!")

if __name__ == "__main__":
    run_ingestion()
