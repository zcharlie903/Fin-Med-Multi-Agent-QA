from __future__ import annotations
import argparse, pathlib
from typing import List, Dict
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.rag.vectorstore import get_vectorstore

def ingest(paths: List[str], domain: str = "medical", collection: str | None = None, chunk_size=1200, chunk_overlap=150):
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    for p in paths:
        pth = pathlib.Path(p)
        if not pth.exists():
            print(f"[skip] not found: {pth}")
            continue
        loader = TextLoader(str(pth), encoding="utf-8")
        raw_docs = loader.load()
        for d in raw_docs:
            d.metadata = {"source": str(pth), "domain": domain}
        docs.extend(splitter.split_documents(raw_docs))

    vs = get_vectorstore(collection)
    if docs:
        vs.add_documents(docs)
    print(f"Ingested {len(docs)} chunks into collection '{vs.collection_name}'.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", required=True)
    ap.add_argument("--domain", default="medical", choices=["medical", "finance"])
    ap.add_argument("--collection", default=None)
    args = ap.parse_args()
    ingest(args.paths, domain=args.domain, collection=args.collection)
