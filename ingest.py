"""Load every PDF in data/, split into chunks, embed, and store in ChromaDB.

Usage: python ingest.py [--reset]
"""
import argparse
import shutil
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

import config
from rag import vector_store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="delete the existing index first")
    args = parser.parse_args()

    if args.reset and config.DB_DIR.exists():
        shutil.rmtree(config.DB_DIR)
        print(f"Removed {config.DB_DIR}")

    pdfs = sorted(config.DATA_DIR.glob("**/*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {config.DATA_DIR}. Drop some in and rerun.")

    pages = []
    for pdf in pdfs:
        loaded = [
            Document(page_content=page.extract_text() or "", metadata={"source": pdf.name, "page": i})
            for i, page in enumerate(PdfReader(pdf).pages)
        ]
        pages.extend(loaded)
        print(f"Loaded {pdf.name}: {len(loaded)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(pages)
    # Stable ids so re-running ingest updates chunks instead of duplicating them.
    ids = [f"{c.metadata['source']}:{c.metadata.get('page', 0)}:{i}" for i, c in enumerate(chunks)]

    print(f"Embedding {len(chunks)} chunks with {config.EMBED_MODEL}...")
    vector_store().add_documents(chunks, ids=ids)
    print(f"Done. Index stored in {config.DB_DIR}")


if __name__ == "__main__":
    main()
