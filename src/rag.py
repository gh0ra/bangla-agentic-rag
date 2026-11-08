import sys
import ollama
import chromadb

LLM = "qwen2.5:7b"
EMB = "bge-m3"

client = chromadb.PersistentClient(path="chroma_db")
col = client.get_collection("bn_wiki")


def retrieve(question, k=4):
    q = ollama.embed(model=EMB, input=question)["embeddings"][0]
    res = col.query(query_embeddings=[q], n_results=k)
    return res["documents"][0], res["metadatas"][0]


def answer(question):
    docs, metas = retrieve(question)
    context = "\n\n".join(f"[{m['title']}] {d}" for d, m in zip(docs, metas))
    prompt = (
        "তুমি একজন সহায়ক। শুধুমাত্র নিচের তথ্য ব্যবহার করে বাংলায় উত্তর দাও। "
        "উত্তর না পেলে বলো 'তথ্য পাওয়া যায়নি'।\n\n"
        f"তথ্য:\n{context}\n\nপ্রশ্ন: {question}\nউত্তর:"
    )
    res = ollama.chat(model=LLM, messages=[{"role": "user", "content": prompt}])
    sources = sorted({m["title"] for m in metas})
    return res["message"]["content"], sources


if __name__ == "__main__":
    q = " ".join(sys.argv[1:])
    a, s = answer(q)
    print(a)
    print("\nসূত্র:", ", ".join(s))