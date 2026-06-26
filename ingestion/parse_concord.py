import re
import logging
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BOOK_MAPPING = {
    "Augsburg Confession": "AC",
    "Apology": "Ap",
    "Smalcald Articles": "SA",
    "Large Catechism": "LC",
    "Small Catechism": "SC",
    "Formula of Concord": "FC",
}

def parse_html_to_chunks(html_content: str, book_name: str) -> list[dict]:
    """
    Parse Book of Concord HTML content into structured chunks containing book metadata,
    article identification, paragraph number, cleaned text, and formatted citations.
    """
    if book_name in BOOK_MAPPING:
        book_abbrev = BOOK_MAPPING[book_name]
    else:
        book_abbrev = "".join(word[0].upper() for word in book_name.split() if word)

    soup = BeautifulSoup(html_content, "html.parser")
    chunks = []
    
    current_article_num = "Unknown"
    current_article_id = f"{book_abbrev}_Unknown"

    for tag in soup.find_all(["h2", "p"]):
        if tag.name == "h2":
            text_content = tag.get_text().strip()
            match = re.search(r"Article\s+([IVXLCDM\d]+)", text_content, re.IGNORECASE)
            if match:
                current_article_num = match.group(1)
                current_article_id = f"{book_abbrev}_{current_article_num}"
        elif tag.name == "p":
            text_content = tag.get_text().strip()
            match = re.match(r"^(\d+)\.\s*(.*)$", text_content, re.DOTALL)
            if match:
                paragraph_number = int(match.group(1))
                para_text = match.group(2).strip()
                
                if current_article_num == "Unknown":
                    logger.warning("Paragraph %d parsed before any valid Article header in book %s", paragraph_number, book_name)
                
                citation = f"{book_name}, Article {current_article_num}, Paragraph {paragraph_number}"
                
                chunks.append({
                    "book": book_name,
                    "article_id": current_article_id,
                    "paragraph_number": paragraph_number,
                    "text": para_text,
                    "citation": citation
                })
            else:
                logger.debug("Skipping unnumbered paragraph in %s: %r", book_name, text_content)
                
    return chunks
