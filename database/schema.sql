CREATE TABLE IF NOT EXISTS book (
    book_id INTEGER PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    testament VARCHAR(2) NOT NULL CHECK (testament IN ('OT', 'NT'))
);

CREATE TABLE IF NOT EXISTS verse (
    verse_id INTEGER PRIMARY KEY,
    book_id INTEGER REFERENCES book(book_id),
    chapter INTEGER NOT NULL,
    verse_number INTEGER NOT NULL,
    original_verse TEXT NOT NULL,
    address_code VARCHAR(50) UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS verse_translation (
    verse_id INTEGER REFERENCES verse(verse_id),
    version_code VARCHAR(10) NOT NULL,
    text TEXT NOT NULL,
    PRIMARY KEY (verse_id, version_code)
);

CREATE INDEX IF NOT EXISTS idx_verse_book ON verse(book_id);

