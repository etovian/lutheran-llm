import os
import time
import logging
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BIBLE_FILES = [
    "https://raw.githubusercontent.com/scrollmapper/bible_databases/master/formats/json/KJV.json",
    "https://raw.githubusercontent.com/scrollmapper/bible_databases/master/formats/json/MKJV.json",
    "https://raw.githubusercontent.com/seven1m/open-bibles/master/eng-web.usfx.xml"
]

BOC_MAP = {
    "apology": ["_index.md", "conclusion.md", "melanchthon-greetings.md", "the-apology.md"],
    "augsburg-confession": ["_index.md", "articles.md", "conclusion.md", "preface.md"],
    "ecumenical-creeds": ["_index.md", "apostles-creed.md", "nicene-creed.md", "athanasian-creed.md"],
    "formula": ["_index.md"],
    "formula/epitome": [
        "_index.md", "church-rites.md", "descent-into-hell.md", "election.md",
        "free-will.md", "good-works.md", "law-and-gospel.md", "original-sin.md",
        "other-sects.md", "person-of-christ.md", "righteousness-of-faith.md",
        "rule-and-norm.md", "the-lords-supper.md", "third-use-law.md"
    ],
    "formula/solid-declaration": [
        "_index.md", "christs-descent-into-hell.md", "church-rites-adiaphora.md",
        "election.md", "free-will.md", "good-works.md", "holy-supper.md",
        "law-gospel.md", "original-sin.md", "other-sects.md", "person-of-christ.md",
        "preface.md", "righteousness-of-faith.md", "rule-and-norm.md", "third-use-law.md"
    ],
    "large-catechism": [
        "_index.md", "apostles-creed.md", "holy-baptism.md", "preface.md",
        "sacrament-of-the-altar.md", "ten-commandments.md", "the-lords-prayer.md"
    ],
    "power-and-primacy": ["_index.md", "power-jurisdiction-bishops.md", "signatories.md", "treatise.md"],
    "smalcald-articles": ["_index.md", "articles.md", "preface.md", "signatories.md"],
    "small-catechism": [
        "_index.md", "baptism.md", "confession-absolution.md", "creed.md",
        "daily-prayers.md", "holy-communion.md", "preface.md",
        "questions-and-answers.md", "table-of-duties.md", "ten-commandments.md", "the-lords-prayer.md"
    ]
}

BOC_BASE_URL = "https://raw.githubusercontent.com/remysheppard/lutheran-confessions/main/content/boc/"
BOC_FILES = []
for directory, files in BOC_MAP.items():
    for f in files:
        url = f"{BOC_BASE_URL}{directory}/{f}"
        BOC_FILES.append((url, f"boc/{directory}/{f}"))

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def download_file(url: str, dest_path: str, max_retries: int = 3, backoff_factor: float = 0.5):
    """Downloads a file from url and saves it to dest_path with retries and backoff."""
    parent_dir = os.path.dirname(dest_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Downloading {url} (attempt {attempt + 1}/{max_retries + 1})")
            response = requests.get(url, headers=DEFAULT_HEADERS, timeout=30)
            response.raise_for_status()
            
            with open(dest_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            
            logger.info(f"Successfully downloaded to {dest_path}")
            return
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"Failed to download {url} after {max_retries} retries: {e}")
                raise e
            sleep_time = backoff_factor * (2 ** attempt)
            logger.warning(f"Error downloading {url}: {e}. Retrying in {sleep_time}s...")
            time.sleep(sleep_time)

def download_all(dest_dir: str = "data"):
    """Downloads all Bible and Book of Concord files to the dest_dir."""
    logger.info(f"Starting ingestion download to cache directory: {dest_dir}")
    
    # 1. Download Bible files
    for url in BIBLE_FILES:
        filename = url.split("/")[-1]
        dest_path = os.path.join(dest_dir, filename)
        download_file(url, dest_path)
        
    # 2. Download Book of Concord files
    for url, rel_path in BOC_FILES:
        # rel_path contains forward slashes, replace with OS specific separator
        parts = rel_path.split("/")
        dest_path = os.path.join(dest_dir, *parts)
        download_file(url, dest_path)
        
    logger.info("All files downloaded successfully.")

if __name__ == "__main__":
    download_all()
