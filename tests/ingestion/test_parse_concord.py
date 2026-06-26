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
