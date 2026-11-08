import os
import requests
import ollama
import chromadb

TITLES = [
    "বাংলাদেশ", "ঢাকা", "বাংলা ভাষা", "রবীন্দ্রনাথ ঠাকুর",
    "কাজী নজরুল ইসলাম", "বাংলাদেশের মুক্তিযুদ্ধ", "সুন্দরবন",
    "পদ্মা সেতু", "কক্সবাজার", "চট্টগ্রাম",
]
API = "https://bn.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "bangla-agentic-rag/0.1 (student project)"}


def fetch(title):
    params = {
        "action": "query", "format": "json", "prop": "extracts",
        "explaintext": 1, "redirects": 1, "titles": title,
    }
    r = requests.get(API, params=params, headers=HEADERS, timeout=30)
    page = next(iter(r.json()["query"]["pages"].values()))
    return page.get("extract", "")


def chunk(text, size=600, overlap=100):
    chunks, start = [], 0
    while start < len(text):
        piece = text[start:start + size].strip()
        if len(piece) > 100:
            chunks.append(piece)
        start += size - overlap
    return chunks


def main():
    os.makedirs("data", exist_ok=True)
    client = chromadb.PersistentClient(path="chroma_db")
    col = client.get_or_create_collection("bn_wiki", metadata={"hnsw:space": "cosine"})

    for title in TITLES:
        text = fetch(title)
        if not text:
            print("skipped:", title)
            continue
        with open(f"data/{title}.txt", "w", encoding="utf-8") as f:
            f.write(text)

        chunks = chunk(text)
        for i in range(0, len(chunks), 32):
            batch = chunks[i:i + 32]
            embs = ollama.embed(model="bge-m3", input=batch)["embeddings"]
            col.upsert(
                ids=[f"{title}-{i + j}" for j in range(len(batch))],
                documents=batch,
                embeddings=embs,
                metadatas=[{"title": title}] * len(batch),
            )
        print(title, len(chunks), "chunks")

    print("total:", col.count())


if __name__ == "__main__":
    main()