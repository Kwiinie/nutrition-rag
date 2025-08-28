import os
import re
import json
import time
from uuid import uuid4
from typing import Optional, List, Literal, Dict, Any, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from openai import OpenAI
from pinecone import Pinecone

# ================= Config =================
load_dotenv()

INDEX_NAME = os.getenv("INDEX_NAME", "nutritiondb")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
INDEX_DIM = int(os.getenv("INDEX_DIM", "1536"))
NAMESPACE = os.getenv("NAMESPACE", "default")

SIMILARITY_THRESHOLD_PRIMARY = float(os.getenv("SIM_PRIMARY", "0.62"))
SIMILARITY_THRESHOLD_FALLBACK = float(os.getenv("SIM_FALLBACK", "0.58"))
TOP_K = int(os.getenv("TOP_K", "8"))
RERANK_TOP = int(os.getenv("RERANK_TOP", "5"))
STRICT_DISEASE_MATCH = os.getenv("STRICT_DISEASE_MATCH", "true").lower() == "true"
MIN_SNIPPETS_TO_ANSWER = int(os.getenv("MIN_SNIPPETS_TO_ANSWER", "2"))
HIGH_SCORE_FOR_GENERAL = float(os.getenv("HIGH_SCORE_FOR_GENERAL", "0.70"))
GENERAL_DISEASE_TAGS = {"", "general", "common"}
CHAT_MODEL = os.getenv("OPENAI_API_MODEL", "gpt-4o-mini")
# ==========================================

# ---------- Normalize & Aliases ----------
try:
    from unidecode import unidecode
except ImportError:
    def unidecode(x): return x

def normalize_text(s: str) -> str:
    s = unidecode((s or "").lower()).strip()
    s = re.sub(r"\s+", " ", s)
    return s

# Disease aliases (multilingual)
DISEASE_ALIASES = {
    "tiểu đường": [
        "tieu duong", "dai thao duong", "type 2", "type 1", "duong huyet",
        "diabetes", "type ii", "type i", "blood sugar", "hyperglycemia", "insulin"
    ],
    "cao huyết áp": [
        "cao huyet ap", "tang huyet ap", "benh huyet ap",
        "hypertension", "high blood pressure", "bp high"
    ],
    "dạ dày": [
        "da day", "viem loet da day", "dau da day", "trao nguoc da day", "bao tu",
        "stomach", "gastritis", "gerd", "acid reflux", "peptic ulcer", "reflux"
    ],
    "gan nhiễm mỡ": [
        "gan nhiem mo", "gan mo", "gan do mo", "gan thoai hoa mo",
        "fatty liver", "nafld", "hepatic steatosis"
    ],
}
SUPPORTED_DISEASES = list(DISEASE_ALIASES.keys())

def match_supported_disease(text: str) -> Optional[str]:
    """Detect if user query mentions a supported disease (using aliases)."""
    qn = normalize_text(text)
    for canonical, aliases in DISEASE_ALIASES.items():
        cands = [normalize_text(canonical)] + [normalize_text(a) for a in aliases]
        if any(c in qn for c in cands):
            return canonical
    return None

def detect_user_language(text: str) -> str:
    """Detect user language (Vietnamese, English, or auto)."""
    t = text or ""
    if re.search(r"[ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]", t, re.IGNORECASE):
        return "vi"
    ascii_only = all(ord(ch) < 128 for ch in t)
    en_markers = ["should i eat", "what to eat", "meal plan", "can i eat",
                  "diet for", "avoid", "hypertension", "diabetes", "stomach", "fatty liver", "gerd"]
    if ascii_only and any(m in t.lower() for m in en_markers):
        return "en"
    return "auto"

# ---------- Load env & init clients ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in .env")
if not PINECONE_API_KEY:
    raise RuntimeError("Missing PINECONE_API_KEY in .env")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

def get_index():
    """Always return a fresh Pinecone index handle (avoids stale handles)."""
    return pc.Index(INDEX_NAME)

# ---------- FastAPI ----------
app = FastAPI(
    title="Nutrition RAG API",
    version="1.4.0",
    description="RAG-first nutrition assistant (OpenAI + Pinecone). Swagger UI at /docs",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ---------- Schemas ----------
class ChatRequest(BaseModel):
    message: str = Field(..., example="Tôi bị dạ dày (GERD), nên ăn gì?")

class ChatResponse(BaseModel):
    response: str
    type: Literal["rag", "general"]
    disease: Optional[str] = None
    notice: Optional[str] = None
    language: Optional[str] = None
    sources: Optional[List[str]] = None

class DiseasesResponse(BaseModel):
    supported_diseases: List[str]

class HealthResponse(BaseModel):
    status: str = "healthy"

# ---------- Retrieval helpers ----------
def get_embeddings(batch_texts: List[str]) -> List[List[float]]:
    """Get embeddings from OpenAI for a list of texts."""
    resp = openai_client.embeddings.create(model=OPENAI_EMBED_MODEL, input=batch_texts)
    out = [d.embedding for d in resp.data]
    for vec in out:
        if len(vec) != INDEX_DIM:
            raise RuntimeError(f"Embedding dimension {len(vec)} != index dimension {INDEX_DIM}")
    return out

def pinecone_query_by_vector(vec: List[float], top_k: int = TOP_K):
    """Query Pinecone with a given embedding vector."""
    res = get_index().query(vector=vec, top_k=top_k, include_metadata=True, namespace=NAMESPACE)
    if hasattr(res, "to_dict"):
        return res.to_dict().get("matches") or []
    return res.get("matches") if isinstance(res, dict) else getattr(res, "matches", [])

def hyde_expand(user_query: str) -> str:
    """Generate synthetic doc (HyDE) to expand search space."""
    p = f"Viết đoạn tóm tắt thực tế (5-7 câu) trả lời ngắn gọn cho câu hỏi: {user_query}"
    r = openai_client.chat.completions.create(
        model=CHAT_MODEL, temperature=0.2, max_tokens=180,
        messages=[
            {"role": "system", "content": "Viết tiếng Việt ngắn gọn. Nếu câu hỏi là tiếng Anh, trả lời tiếng Anh."},
            {"role": "user", "content": p}
        ]
    )
    return (r.choices[0].message.content or "").strip()

def expand_queries(user_query: str, max_extra: int = 2) -> List[str]:
    """Generate paraphrases of the query for recall boost."""
    prompt = f"""
Give {max_extra} alternative short paraphrases (≤15 words) for the question below, one per line.
Question: {user_query}
"""
    try:
        r = openai_client.chat.completions.create(
            model=CHAT_MODEL, temperature=0.3, max_tokens=120,
            messages=[
                {"role": "system", "content": "Return only the lines, no explanation."},
                {"role": "user", "content": prompt}
            ]
        ).choices[0].message.content.strip().splitlines()
        return [s.strip(" -•").strip() for s in r if s.strip()][:max_extra]
    except Exception:
        return []

def merge_matches(all_matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge Pinecone matches by id, keep highest score."""
    by_id: Dict[str, Dict[str, Any]] = {}
    for m in all_matches:
        mid = (m.get("id") if isinstance(m, dict) else getattr(m, "id", None)) or ""
        score = m.get("score") if isinstance(m, dict) else getattr(m, "score", 0.0)
        meta = m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", {}) or {}
        if not mid:
            continue
        if mid not in by_id or score > by_id[mid]["score"]:
            by_id[mid] = {"id": mid, "score": float(score or 0.0), "metadata": meta}
    merged = list(by_id.values())
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged

def pick_context_blocks(merged: List[Dict[str, Any]], disease: Optional[str]) -> Tuple[List[Dict[str, Any]], str]:
    """Select the most relevant context snippets for RAG."""
    wanted = normalize_text(disease) if disease else None

    def md_disease(md: Dict[str, Any]) -> str:
        return normalize_text((md or {}).get("disease") or "")

    def filter_by_score(items, thr):
        return [m for m in items if m["score"] >= thr]

    if disease:
        same = [m for m in merged if md_disease(m["metadata"]) == wanted]
        primary = filter_by_score(same, SIMILARITY_THRESHOLD_PRIMARY)
        if len(primary) >= MIN_SNIPPETS_TO_ANSWER:
            return primary[:RERANK_TOP], "rag"
        fallback = filter_by_score(same, SIMILARITY_THRESHOLD_FALLBACK)
        if len(fallback) >= MIN_SNIPPETS_TO_ANSWER:
            return fallback[:RERANK_TOP], "rag"
        return [], "none"

    # no disease detected
    generalish = [m for m in merged if md_disease(m["metadata"]) in GENERAL_DISEASE_TAGS]
    strong = filter_by_score(generalish, HIGH_SCORE_FOR_GENERAL)
    if len(strong) >= MIN_SNIPPETS_TO_ANSWER:
        return strong[:RERANK_TOP], "rag"
    return [], "none"

# ---------- LLM Answer ----------
def chat_complete(prompt: str, user_query: str, user_lang: str) -> str:
    """Ask LLM to generate the final answer."""
    lang_hint = {
        "vi": "Hãy trả lời hoàn toàn bằng TIẾNG VIỆT.",
        "en": "Answer entirely in ENGLISH.",
        "auto": "Answer in the user's language (detect automatically)."
    }.get(user_lang, "Answer in the user's language (detect automatically).")
    sys = "You are a board-certified dietitian-nutritionist.\n" + lang_hint
    r = openai_client.chat.completions.create(
        model=CHAT_MODEL, temperature=0.7, max_tokens=900,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": f"User query: {user_query}\n\n{prompt}"}
        ]
    )
    return r.choices[0].message.content

def build_answer_prompt(contexts: List[Dict[str, Any]], user_query: str, disease: Optional[str]) -> Tuple[str, List[str]]:
    """Build answer prompt from selected context snippets."""
    texts, sources = [], []
    for c in contexts:
        md = c.get("metadata") or {}
        txt = md.get("full_text") or md.get("content") or md.get("text") or ""
        if not txt: continue
        texts.append(f"- {txt}")
        src = md.get("source") or md.get("title") or ""
        if src: sources.append(src)
    ctx_text = "\n\n".join(texts)
    sources = list(dict.fromkeys(sources))
    prompt = f"""
From the RAG knowledge base:
{ctx_text}

User question: {user_query}
"""
    return prompt, sources

# ---------- RAG-first Pipeline ----------
def rag_first_or_general(user_query: str, user_lang: str) -> Tuple[str, str, Optional[str], List[str]]:
    """Main pipeline: try RAG first, fallback to general if no context."""
    disease = match_supported_disease(user_query)
    variants: List[str] = [user_query]
    try:
        variants += expand_queries(user_query, 2)
        variants.append(hyde_expand(user_query))
    except Exception: pass

    all_matches: List[Dict[str, Any]] = []
    try:
        vecs = get_embeddings(variants)
        for vec in vecs:
            all_matches.extend(pinecone_query_by_vector(vec, top_k=TOP_K))
    except Exception: pass

    merged = merge_matches(all_matches)
    if not merged:
        return get_general_response(user_query, user_lang), "general", disease, []

    selected, mode = pick_context_blocks(merged, disease)
    if mode == "none":
        return get_general_response(user_query, user_lang), "general", disease, []

    prompt, sources = build_answer_prompt(selected, user_query, disease)
    answer = chat_complete(prompt, user_query, user_lang)
    return answer, "rag", disease, sources

# ---------- General fallback ----------
def get_general_response(query: str, user_lang: str) -> str:
    """Generate safe fallback answer when no RAG context available."""
    prompt = f"""
No RAG context found for:
{query}

Give a safe, practical general answer based on standard guidelines.
"""
    return chat_complete(prompt, query, user_lang)

# ---------- Routes ----------
@app.get("/api/health", response_model=HealthResponse, tags=["Meta"])
def health_check():
    return HealthResponse(status="healthy")

@app.get("/api/diseases", response_model=DiseasesResponse, tags=["Meta"])
def get_supported_diseases():
    return DiseasesResponse(supported_diseases=SUPPORTED_DISEASES)

@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
def chat(req: ChatRequest):
    """Main chat endpoint: classify disease, run RAG pipeline or fallback."""
    query = req.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Vui lòng nhập câu hỏi")
    user_lang = detect_user_language(query)
    try:
        answer, mode, disease, sources = rag_first_or_general(query, user_lang)
    except Exception as e:
        print(f"[PIPELINE ERROR] {e}")
        answer, mode, disease, sources = get_general_response(query, user_lang), "general", match_supported_disease(query), []
    notice_map = {"rag": "RAG mode", "general": "No RAG context"}
    return ChatResponse(
        response=answer,
        type=mode,
        disease=disease,
        notice=notice_map.get(mode, mode),
        language=user_lang,
        sources=sources or None
    )

# ---------- Main ----------
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Nutrition RAG API…")
    print("📖 Swagger UI: http://localhost:8000/docs")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
