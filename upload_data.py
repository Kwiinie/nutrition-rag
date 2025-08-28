import os, json, argparse, time
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INDEX_NAME  = os.getenv("INDEX_NAME", "nutritiondb")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")  # 1536 dims
INDEX_DIM   = int(os.getenv("INDEX_DIM", "1536"))
NAMESPACE   = os.getenv("NAMESPACE", "default")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_ENV   = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))


def chunk_text(text: str, size: int = 1000, overlap: int = 100) -> List[str]:
    text = (text or "").strip()
    if not text: return []
    chunks, i, n = [], 0, len(text)
    while i < n:
        j = min(i + size, n)
        ck = text[i:j].strip()
        if ck: chunks.append(ck)
        if j == n: break
        i = max(0, j - overlap)
    return chunks

def main():
    parser = argparse.ArgumentParser(description="Demo: Upload JSONL vào Pinecone")
    parser.add_argument("-f", "--file", default="data.jsonl", help="Đường dẫn JSONL")
    parser.add_argument("-i", "--index", default=INDEX_NAME, help="Tên index")
    parser.add_argument("-n", "--namespace", default=NAMESPACE, help="Namespace")
    parser.add_argument("--no-chunk", action="store_true", help="Không chunk văn bản")
    args = parser.parse_args()

    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    pcone_key  = os.getenv("PINECONE_API_KEY")
    if not openai_key or not pcone_key:
        raise RuntimeError("Mising OPENAI_API_KEY or PINECONE_API_KEY ")

    client = OpenAI(api_key=openai_key)
    pc = Pinecone(api_key=pcone_key)

    # 1) CHECK INDEX
    existing = [x["name"] for x in pc.list_indexes()]
    if args.index not in existing:
        print(f"Creating index '{args.index}'...")
        pc.create_index(
            name=args.index, dimension=INDEX_DIM, metric="cosine",
            spec=ServerlessSpec(cloud=CLOUD, region=REGION)
        )
    print("Waiting for index ready...")
    for _ in range(30):
        if pc.describe_index(args.index).get("status", {}).get("ready"):
            break
        time.sleep(1)
    print("Index ready.")

    index = pc.Index(args.index)

    # 2) READ JSONL
    records: List[Dict[str, Any]] = []
    with open(args.file, "r", encoding="utf-8") as f:
        for ln, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw: continue
            try:
                rec = json.loads(raw)
                records.append(rec)
            except json.JSONDecodeError as e:
                print(f"Line {ln} error JSON: {e}")

    if not records:
        print("No valid record"); return

    # 3) CREATE LIST FOR EMBEDDING
    items = []
    for rec in records:
        rid = str(rec.get("id", f"rec_{len(items)}"))
        disease = str(rec.get("disease", "")).lower()
        txt = rec.get("content") or rec.get("text") or rec.get("full_text") or rec.get("body") or ""
        if not txt: 
            continue
        chunks = [txt] if args.no_chunk else chunk_text(txt, 1000, 100)
        for idx, ck in enumerate(chunks):
            items.append({
                "id": f"{rid}_ck{idx:03d}",
                "text": ck,
                "metadata": {
                    "disease": disease,
                    "recommendations": rec.get("recommendations", "")
                }
            })

    if not items:
        print("No text for embedded"); return

    # 4) EMBEDDING
    vectors = []
    for k in range(0, len(items), BATCH_SIZE):
        batch = items[k:k+BATCH_SIZE]
        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=[it["text"] for it in batch]
        )
        for it, emb in zip(batch, resp.data):
            vectors.append({
                "id": it["id"],
                "values": emb.embedding,
                "metadata": {
                    **it["metadata"],
                    "content": it["text"] 
                }
            })
        print(f"Embedded {min(k+BATCH_SIZE, len(items))}/{len(items)}")

    uploaded = 0
    for k in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[k:k+BATCH_SIZE]
        index.upsert(vectors=batch, namespace=args.namespace)
        uploaded += len(batch)
        print(f" Upsert {uploaded}/{len(vectors)}")

    print("Done.")

if __name__ == "__main__":
    main()
