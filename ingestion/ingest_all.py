import os
import re
import json
import glob
import logging
import xml.etree.ElementTree as ET
from sqlalchemy import text

from config.settings import Settings
from database.connection import get_engine
from ingestion.parse_bible import parse_original_word_tokens
from ingestion.vector_indexer import VectorIndexer
import chromadb

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CANONICAL_BOOKS = [
    (1, "Genesis", "OT", "GEN"),
    (2, "Exodus", "OT", "EXO"),
    (3, "Leviticus", "OT", "LEV"),
    (4, "Numbers", "OT", "NUM"),
    (5, "Deuteronomy", "OT", "DEU"),
    (6, "Joshua", "OT", "JOS"),
    (7, "Judges", "OT", "JDG"),
    (8, "Ruth", "OT", "RUT"),
    (9, "1 Samuel", "OT", "1SA"),
    (10, "2 Samuel", "OT", "2SA"),
    (11, "1 Kings", "OT", "1KI"),
    (12, "2 Kings", "OT", "2KI"),
    (13, "1 Chronicles", "OT", "1CH"),
    (14, "2 Chronicles", "OT", "2CH"),
    (15, "Ezra", "OT", "EZR"),
    (16, "Nehemiah", "OT", "NEH"),
    (17, "Esther", "OT", "EST"),
    (18, "Job", "OT", "JOB"),
    (19, "Psalms", "OT", "PSA"),
    (20, "Proverbs", "OT", "PRO"),
    (21, "Ecclesiastes", "OT", "ECC"),
    (22, "Song of Solomon", "OT", "SNG"),
    (23, "Isaiah", "OT", "ISA"),
    (24, "Jeremiah", "OT", "JER"),
    (25, "Lamentations", "OT", "LAM"),
    (26, "Ezekiel", "OT", "EZK"),
    (27, "Daniel", "OT", "DAN"),
    (28, "Hosea", "OT", "HOS"),
    (29, "Joel", "OT", "JOL"),
    (30, "Amos", "OT", "AMO"),
    (31, "Obadiah", "OT", "OBA"),
    (32, "Jonah", "OT", "JON"),
    (33, "Micah", "OT", "MIC"),
    (34, "Nahum", "OT", "NAM"),
    (35, "Habakkuk", "OT", "HAB"),
    (36, "Zephaniah", "OT", "ZEP"),
    (37, "Haggai", "OT", "HAG"),
    (38, "Zechariah", "OT", "ZEC"),
    (39, "Malachi", "OT", "MAL"),
    (40, "Matthew", "NT", "MAT"),
    (41, "Mark", "NT", "MRK"),
    (42, "Luke", "NT", "LUK"),
    (43, "John", "NT", "JHN"),
    (44, "Acts", "NT", "ACT"),
    (45, "Romans", "NT", "ROM"),
    (46, "1 Corinthians", "NT", "1CO"),
    (47, "2 Corinthians", "NT", "2CO"),
    (48, "Galatians", "NT", "GAL"),
    (49, "Ephesians", "NT", "EPH"),
    (50, "Philippians", "NT", "PHP"),
    (51, "Colossians", "NT", "COL"),
    (52, "1 Thessalonians", "NT", "1TH"),
    (53, "2 Thessalonians", "NT", "2TH"),
    (54, "1 Timothy", "NT", "1TI"),
    (55, "2 Timothy", "NT", "2TI"),
    (56, "Titus", "NT", "TIT"),
    (57, "Philemon", "NT", "PHM"),
    (58, "Hebrews", "NT", "HEB"),
    (59, "James", "NT", "JAS"),
    (60, "1 Peter", "NT", "1PE"),
    (61, "2 Peter", "NT", "2PE"),
    (62, "1 John", "NT", "1JN"),
    (63, "2 John", "NT", "2JN"),
    (64, "3 John", "NT", "3JN"),
    (65, "Jude", "NT", "JUD"),
    (66, "Revelation of John", "NT", "REV")
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
    {"strongs_number": "G2744", "pronunciation": "kow-khah'-om-ahee", "definition": "to boast, glory, exult", "derivation": "from αυχην"},
    # Added definitions
    {"strongs_number": "G907", "pronunciation": "bap-tid'-zo", "definition": "to baptize, dip, wash", "derivation": "from a derivative of G911"},
    {"strongs_number": "G3067", "pronunciation": "loo-tron'", "definition": "bathing, washing, laver", "derivation": "from G3068"},
    {"strongs_number": "G3824", "pronunciation": "pal-ing-ghen-es-ee'-ah", "definition": "rebirth, regeneration, renovation", "derivation": "from G3825 and G1078"},
    {"strongs_number": "G4983", "pronunciation": "so'-mah", "definition": "body", "derivation": "from G4982"},
    {"strongs_number": "G2842", "pronunciation": "koy-nohn-ee'-ah", "definition": "fellowship, communion, sharing", "derivation": "from G2844"},
    {"strongs_number": "G1722", "pronunciation": "en", "definition": "in, with, by, among", "derivation": "a primary preposition"}
]

KEY_THEOLOGICAL_VERSES = {
    "ROM_3_28": "λογιζόμεθα[G3049] οὖν[G3767] πίστει[G4102] δικαιοῦσθαι[G1344] ἄνθρωπον[G444] χωρὶς[G5565] ἔργων[G2041] νόμου[G3551]",
    "EPH_2_8": "τῇ[G3588] γὰρ[G1063] χάριτί[G5485] ἐστε[G2075] σεσωσμένοι[G4982] διὰ[G1223] πίστεως[G4102]",
    "EPH_2_9": "οὐκ[G3756] ἐξ[G1537] ἔργων[G2041] ἵνα[G2443] μή[G3361] τις[G5100] καυχήσηται[G2744]",
    "TIT_3_5": "οὐκ[G3756] ἐξ[G1537] ἔργων[G2041] τῶν[G3588] ἐν[G1722] δικαιοσύνῃ[G1343] ἃ[G3739] ἐποιήσαμεν[G4160] ἡμεῖς[G2249] ἀλλὰ[G235] κατὰ[G2596] τὸ[G3588] αὐτοῦ[G846] ἔλεος[G1656] ἔσωσεν[G4982] ἡμᾶς[G2248] διὰ[G1223] λουτροῦ[G3067] παλιγγενεσίας[G3824] καὶ[G2532] ἀνακαινώσεως[G341] πνεύματος[G4151] ἁγίου[G40]",
    "1CO_11_24": "καὶ[G2532] εὐχαριστήσας[G2168] ἔκλασεν[G2806] καὶ[G2532] εἶπεν[G2036] λάβετε[G2983] φάγετε[G5315] τοῦτό[G5124] μού[G3450] ἐστιν[G2070] τὸ[G3588] σῶμα[G4983] τὸ[G3588] ὑπὲρ[G5228] ὑμῶν[G5216] κλώμενον[G2806] τοῦτο[G5124] ποιεῖτε[G4160] εἰς[G1519] τὴν[G3588] ἐμὴν[G1699] ἀνάμνησιν[G364]"
}

DIR_TO_BOOK = {
    "apology": "Apology",
    "augsburg-confession": "Augsburg Confession",
    "ecumenical-creeds": "Ecumenical Creeds",
    "formula": "Formula of Concord",
    "formula/epitome": "Formula of Concord",
    "formula/solid-declaration": "Formula of Concord",
    "large-catechism": "Large Catechism",
    "power-and-primacy": "Power and Primacy",
    "smalcald-articles": "Smalcald Articles",
    "small-catechism": "Small Catechism"
}

BOOK_ABBREVIATIONS = {
    "Augsburg Confession": "AC",
    "Apology": "Ap",
    "Smalcald Articles": "SA",
    "Large Catechism": "LC",
    "Small Catechism": "SC",
    "Formula of Concord": "FC",
    "Power and Primacy": "PP",
    "Ecumenical Creeds": "EC"
}

def clear_tables(conn):
    logger.info("Clearing relational database tables...")
    conn.execute(text("DELETE FROM original_word"))
    conn.execute(text("DELETE FROM verse_translation"))
    conn.execute(text("DELETE FROM verse"))
    conn.execute(text("DELETE FROM book"))
    conn.execute(text("DELETE FROM strongs_concordance"))

def seed_books(conn):
    logger.info("Seeding canonical books...")
    for book_id, name, testament, _ in CANONICAL_BOOKS:
        conn.execute(
            text("INSERT INTO book (book_id, name, testament) VALUES (:book_id, :name, :testament)"),
            {"book_id": book_id, "name": name, "testament": testament}
        )

def seed_strongs(conn):
    logger.info("Seeding Strong's concordance definitions...")
    for st in STRONGS_CONCORDANCE:
        conn.execute(
            text("INSERT INTO strongs_concordance (strongs_number, pronunciation, definition, derivation) "
                 "VALUES (:strongs_number, :pronunciation, :definition, :derivation)"),
            st
        )

def get_tag_name(tag: str) -> str:
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag

def parse_usfx_xml(xml_content_or_path: str, conn, key_verses: dict) -> dict:
    logger.info("Parsing USFX XML file...")
    if xml_content_or_path.strip().startswith("<"):
        root = ET.fromstring(xml_content_or_path)
    else:
        tree = ET.parse(xml_content_or_path)
        root = tree.getroot()

    allowed_usfx_ids = {b[3] for b in CANONICAL_BOOKS}
    # Build maps
    book_elements = {}
    for book_el in root.findall(".//book"):
        b_id = book_el.attrib.get("id")
        if b_id in allowed_usfx_ids:
            book_elements[b_id] = book_el
            
    address_to_id = {}
    verse_id_counter = 1
    
    parsed_verses = []
    
    for book_id_int, book_name, testament, usfx_id in CANONICAL_BOOKS:
        book_tag = book_elements.get(usfx_id)
        if book_tag is None:
            continue
            
        current_chapter = None
        current_verse_num = None
        current_verse_text_parts = []
        
        def save_current_verse():
            nonlocal current_verse_num, current_verse_text_parts, verse_id_counter
            if current_verse_num is not None and current_chapter is not None:
                verse_text = "".join(current_verse_text_parts).strip()
                verse_text = " ".join(verse_text.split())
                
                address_code = f"{usfx_id}_{current_chapter}_{current_verse_num}"
                orig_verse = key_verses.get(address_code, "")
                
                parsed_verses.append({
                    "verse_id": verse_id_counter,
                    "book_id": book_id_int,
                    "chapter": current_chapter,
                    "verse_number": current_verse_num,
                    "original_verse": orig_verse,
                    "address_code": address_code,
                    "text": verse_text
                })
                address_to_id[address_code] = verse_id_counter
                verse_id_counter += 1
                
                current_verse_num = None
                current_verse_text_parts = []

        def traverse(el):
            nonlocal current_chapter, current_verse_num
            tag = get_tag_name(el.tag)
            
            if tag == "c":
                save_current_verse()
                c_id = el.attrib.get("id")
                if c_id and c_id.isdigit():
                    current_chapter = int(c_id)
            elif tag == "v":
                save_current_verse()
                v_id = el.attrib.get("id")
                if v_id and v_id.isdigit():
                    current_verse_num = int(v_id)
            elif tag == "ve":
                save_current_verse()
                current_verse_num = None
                
            if tag not in ("f", "x"):
                if current_verse_num is not None and el.text:
                    current_verse_text_parts.append(el.text)
                for child in el:
                    traverse(child)
                    
            if current_verse_num is not None and el.tail:
                current_verse_text_parts.append(el.tail)

        # Traverse book tag children
        traverse(book_tag)
        save_current_verse()
        
    # Bulk insert verses and translations in batches to prevent potential memory or limit issues
    if parsed_verses:
        batch_size = 5000
        logger.info(f"Bulk inserting {len(parsed_verses)} verses into database...")
        for i in range(0, len(parsed_verses), batch_size):
            batch = parsed_verses[i : i + batch_size]
            conn.execute(
                text("INSERT INTO verse (verse_id, book_id, chapter, verse_number, original_verse, address_code) "
                     "VALUES (:verse_id, :book_id, :chapter, :verse_number, :original_verse, :address_code)"),
                [
                    {
                        "verse_id": v["verse_id"],
                        "book_id": v["book_id"],
                        "chapter": v["chapter"],
                        "verse_number": v["verse_number"],
                        "original_verse": v["original_verse"],
                        "address_code": v["address_code"]
                    }
                    for v in batch
                ]
            )
            conn.execute(
                text("INSERT INTO verse_translation (verse_id, version_code, text) "
                     "VALUES (:verse_id, 'WEB', :text)"),
                [
                    {
                        "verse_id": v["verse_id"],
                        "text": v["text"]
                    }
                    for v in batch
                ]
            )
            
        # Parse and bulk-insert original language words
        all_tokens = []
        for v in parsed_verses:
            if v["original_verse"]:
                word_tokens = parse_original_word_tokens(v["original_verse"], verse_id=v["verse_id"])
                all_tokens.extend(word_tokens)
        if all_tokens:
            logger.info(f"Bulk inserting {len(all_tokens)} original language words...")
            conn.execute(
                text("INSERT INTO original_word (verse_id, word_index, word_text, lemma, strongs_number) "
                     "VALUES (:verse_id, :word_index, :word_text, :lemma, :strongs_number)"),
                all_tokens
            )
                    
    logger.info(f"Loaded {len(address_to_id)} canonical verses from USFX XML.")
    return address_to_id

def parse_json_translations(kjv_path: str, mkjv_path: str, conn, address_to_id: dict):
    logger.info("Parsing alternate translations (KJV and MKJV)...")
    book_name_to_usfx_id = {b[1]: b[3] for b in CANONICAL_BOOKS}
    roman_overrides = {
        "I Samuel": "1SA",
        "II Samuel": "2SA",
        "I Kings": "1KI",
        "II Kings": "2KI",
        "I Chronicles": "1CH",
        "II Chronicles": "2CH",
        "I Corinthians": "1CO",
        "II Corinthians": "2CO",
        "I Thessalonians": "1TH",
        "II Thessalonians": "2TH",
        "I Timothy": "1TI",
        "II Timothy": "2TI",
        "I Peter": "1PE",
        "II Peter": "2PE",
        "I John": "1JN",
        "II John": "2JN",
        "III John": "3JN",
    }
    book_name_to_usfx_id.update(roman_overrides)
    
    for file_path, version_code in [(kjv_path, "KJV"), (mkjv_path, "MKJV")]:
        if not os.path.exists(file_path):
            logger.warning(f"Translation file not found: {file_path}")
            continue
            
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        translations_to_insert = []
        for book in data.get("books", []):
            book_name = book.get("name")
            usfx_id = book_name_to_usfx_id.get(book_name)
            if not usfx_id:
                continue
                
            for chapter in book.get("chapters", []):
                chap_num = chapter.get("chapter")
                for verse in chapter.get("verses", []):
                    verse_num = verse.get("verse")
                    text_val = verse.get("text", "").strip()
                    
                    address_code = f"{usfx_id}_{chap_num}_{verse_num}"
                    verse_id = address_to_id.get(address_code)
                    if verse_id:
                        translations_to_insert.append({
                            "verse_id": verse_id,
                            "version_code": version_code,
                            "text": text_val
                        })
                        
        if translations_to_insert:
            batch_size = 5000
            logger.info(f"Bulk inserting {len(translations_to_insert)} translations for {version_code}...")
            for i in range(0, len(translations_to_insert), batch_size):
                batch = translations_to_insert[i : i + batch_size]
                conn.execute(
                    text("INSERT INTO verse_translation (verse_id, version_code, text) "
                         "VALUES (:verse_id, :version_code, :text)"),
                    batch
                )
        logger.info(f"Loaded {len(translations_to_insert)} translation verses for {version_code}.")

def strip_yaml_frontmatter(content: str) -> str:
    lines = content.splitlines()
    if lines and lines[0].strip() == "---":
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                return "\n".join(lines[idx+1:])
    return content

def parse_boc_markdown(boc_dir: str) -> list[dict]:
    logger.info("Parsing Book of Concord markdown files...")
    md_files = glob.glob(os.path.join(boc_dir, "**", "*.md"), recursive=True)
    chunks = []
    md_files.sort()
    
    # Sort DIR_TO_BOOK keys by length descending to match longest path prefixes first
    sorted_dir_keys = sorted(DIR_TO_BOOK.keys(), key=len, reverse=True)
    
    for file_path in md_files:
        filename = os.path.basename(file_path)
        if filename.startswith("_") or filename.lower() in ["index.md", "navigation.md"]:
            continue
            
        rel_path = os.path.relpath(file_path, boc_dir).replace("\\", "/")
        book_name = "Unknown"
        for key in sorted_dir_keys:
            if rel_path.startswith(key):
                book_name = DIR_TO_BOOK[key]
                break
                
        book_abbrev = BOOK_ABBREVIATIONS.get(book_name, "".join(w[0].upper() for w in book_name.split() if w))
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        content = strip_yaml_frontmatter(content)
        lines = content.splitlines()
        
        file_slug = os.path.splitext(filename)[0]
        current_heading = file_slug.replace("-", " ").title()
        current_section_lines = []
        sections = []
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                if current_section_lines:
                    sections.append((current_heading, "\n".join(current_section_lines)))
                    current_section_lines = []
                current_heading = stripped.lstrip("#").strip()
            else:
                current_section_lines.append(line)
                
        if current_section_lines:
            sections.append((current_heading, "\n".join(current_section_lines)))
            
        for heading, section_text in sections:
            if heading.lower() in ["index", "navigation", "table of contents", "index or navigation"]:
                continue
                
            roman_match = re.search(r"\b([IVXLCDM]+)\b", heading)
            if roman_match:
                article_id = f"{book_abbrev}_{roman_match.group(1)}"
            else:
                heading_slug = re.sub(r"[^a-zA-Z0-9]+", "_", heading).strip("_")
                if heading_slug:
                    article_id = f"{book_abbrev}_{heading_slug}"
                else:
                    article_id = f"{book_abbrev}_{file_slug.replace('-', '_')}"
                    
            raw_paragraphs = section_text.split("\n\n")
            paragraph_number = 1
            
            for p in raw_paragraphs:
                cleaned_p = p.strip()
                if not cleaned_p:
                    continue
                cleaned_p = " ".join(cleaned_p.split())
                if not cleaned_p:
                    continue
                    
                if ".md" in cleaned_p.lower() and "[" in cleaned_p and "](" in cleaned_p:
                    continue
                    
                citation = f"{book_name}, {heading}, Paragraph {paragraph_number}"
                chunks.append({
                    "text": cleaned_p,
                    "book": book_name,
                    "article_id": article_id,
                    "paragraph_number": paragraph_number,
                    "citation": citation
                })
                paragraph_number += 1
                
    logger.info(f"Parsed {len(chunks)} paragraphs/chunks from Book of Concord markdown.")
    return chunks

def run_ingest_all(
    usfx_path: str = None,
    kjv_path: str = None,
    mkjv_path: str = None,
    boc_dir: str = None,
    settings: Settings = None
):
    if settings is None:
        settings = Settings()
    
    if usfx_path is None:
        usfx_path = os.path.join("data", "eng-web.usfx.xml")
    if kjv_path is None:
        kjv_path = os.path.join("data", "KJV.json")
    if mkjv_path is None:
        mkjv_path = os.path.join("data", "MKJV.json")
    if boc_dir is None:
        boc_dir = os.path.join("data", "boc")

    engine = get_engine(settings)
    
    # 1. Database seeding
    logger.info("Initializing relational database tables...")
    with engine.begin() as conn:
        clear_tables(conn)
        seed_books(conn)
        seed_strongs(conn)
        
        # Parse and insert Bible verses (WEB)
        address_to_id = parse_usfx_xml(usfx_path, conn, KEY_THEOLOGICAL_VERSES)
        
        # Parse and insert KJV / MKJV translations
        parse_json_translations(kjv_path, mkjv_path, conn, address_to_id)
        
    # 2. Parse Book of Concord markdown chunks
    boc_chunks = parse_boc_markdown(boc_dir)
    
    # 3. ChromaDB collection recreate & indexing
    logger.info("Initializing ChromaDB client...")
    chroma_client = chromadb.PersistentClient(path=settings.chroma_db_path)
    
    for coll_name in ["confessional_collection", "biblical_collection"]:
        try:
            chroma_client.delete_collection(coll_name)
            logger.info(f"Deleted old ChromaDB collection: {coll_name}")
        except Exception:
            pass
        chroma_client.create_collection(coll_name)
        logger.info(f"Created/Cleared ChromaDB collection: {coll_name}")
        
    indexer = VectorIndexer(chroma_client=chroma_client)
    
    # Index Book of Concord chunks
    logger.info(f"Indexing {len(boc_chunks)} Book of Concord chunks...")
    for i in range(0, len(boc_chunks), 500):
        batch = boc_chunks[i : i + 500]
        indexer.index_confessional_batch(batch)
        logger.info(f"Indexed chunks {i} to {i + len(batch)}...")
        
    # Index NT verses
    logger.info("Fetching New Testament verses from relational DB for indexing...")
    nt_verses = []
    with engine.connect() as conn:
        query = text("""
            SELECT v.verse_id, v.address_code, v.chapter, v.verse_number, b.name as book_name, vt.text
            FROM verse v
            JOIN book b ON v.book_id = b.book_id
            JOIN verse_translation vt ON v.verse_id = vt.verse_id
            WHERE v.book_id >= 40 AND vt.version_code = :version_code
            ORDER BY v.verse_id
        """)
        rows = conn.execute(query, {"version_code": settings.primary_search_version}).mappings().all()
        for r in rows:
            nt_verses.append({
                "verse_id": r["verse_id"],
                "address_code": r["address_code"],
                "chapter": r["chapter"],
                "verse_number": r["verse_number"],
                "book_name": r["book_name"],
                "text": r["text"]
            })
            
    logger.info(f"Indexing {len(nt_verses)} New Testament verses...")
    for i in range(0, len(nt_verses), 1000):
        batch = nt_verses[i : i + 1000]
        indexer.index_biblical_batch(batch)
        logger.info(f"Indexed NT verses {i} to {i + len(batch)}...")
        
    logger.info("All ingestion pipelines completed successfully!")

if __name__ == "__main__":
    run_ingest_all()
