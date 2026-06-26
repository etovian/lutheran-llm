from ingestion.parse_concord import parse_html_to_chunks

def test_parse_concord_paragraphs():
    """Verify that raw HTML is parsed into correct Book of Concord chunks and citations."""
    html_content = """
    <html>
      <body>
        <h1>Augsburg Confession</h1>
        <h2>Article IV. Of Justification</h2>
        <p class="para">1. Also they teach that men cannot be justified before God...</p>
        <p class="para">2. This faith God imputes for righteousness...</p>
      </body>
    </html>
    """
    chunks = parse_html_to_chunks(html_content, book_name="Augsburg Confession")
    
    assert len(chunks) == 2
    assert chunks[0]["book"] == "Augsburg Confession"
    assert chunks[0]["article_id"] == "AC_IV"
    assert chunks[0]["paragraph_number"] == 1
    assert chunks[0]["text"] == "Also they teach that men cannot be justified before God..."
    assert chunks[0]["citation"] == "Augsburg Confession, Article IV, Paragraph 1"
    
    assert chunks[1]["paragraph_number"] == 2
    assert chunks[1]["text"] == "This faith God imputes for righteousness..."
    assert chunks[1]["citation"] == "Augsburg Confession, Article IV, Paragraph 2"


def test_parse_concord_fallback_abbreviation():
    """Verify initials generation for a book name not in the mapping dictionary."""
    chunks = parse_html_to_chunks("<h2>Article I</h2><p>1. Text</p>", "Augsburg Confession Apology")
    assert chunks[0]["article_id"] == "ACA_I"


def test_parse_concord_missing_article_header():
    """Verify fallback behavior ('Unknown') when paragraphs precede article headers."""
    chunks = parse_html_to_chunks("<p>1. Text</p>", "Augsburg Confession")
    assert chunks[0]["article_id"] == "AC_Unknown"
    assert chunks[0]["citation"] == "Augsburg Confession, Article Unknown, Paragraph 1"


def test_parse_concord_skips_unnumbered_paragraphs():
    """Verify that paragraphs without matching number patterns are skipped."""
    html = "<h2>Article I</h2><p>Introductory paragraph without number</p><p>1. Paragraph content</p>"
    chunks = parse_html_to_chunks(html, "Augsburg Confession")
    assert len(chunks) == 1
    assert chunks[0]["paragraph_number"] == 1
