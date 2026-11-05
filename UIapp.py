import os
import time
import shutil
import logging
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from groq import Groq

load_dotenv()
logger = logging.getLogger("uvicorn.error")
groq_client = Groq()   # GROQ_API_KEY .env se leta hai

# ==========================================
# 1. RAG ENGINE INITIALIZATION
# ==========================================

PDF_FILE = "ML(doc).pdf"
PERSIST_DIR = "Chroma_db"

if not os.path.exists(PDF_FILE):
    raise FileNotFoundError(f"Source PDF '{PDF_FILE}' not found in working directory.")

# EMBEDDINGS_BACKEND=fastembed uses a small ONNX model (no torch) so the app fits
# in low-memory hosts such as Render's free tier. Default keeps the original behaviour.
# On Render (which sets RENDER=true) default to the light backend so no env change is needed.
DEFAULT_BACKEND = "fastembed" if os.getenv("RENDER") else "huggingface"
if os.getenv("EMBEDDINGS_BACKEND", DEFAULT_BACKEND).lower() == "fastembed":
    from langchain_community.embeddings import FastEmbedEmbeddings
    embedding_model = FastEmbedEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_dir=os.getenv("FASTEMBED_CACHE", "models"),
    )
else:
    from langchain_huggingface import HuggingFaceEmbeddings
    embedding_model = HuggingFaceEmbeddings()

def _open_existing_index():
    """Reuse Chroma_db when it is usable; otherwise discard it so it gets rebuilt.

    Handles an empty index and an index built with a different embedding model
    (dimension mismatch), which would otherwise break every query with a 500.
    """
    if not (os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR)):
        return None
    store = None
    try:
        store = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)
        if not store.get(limit=1).get("ids"):
            return None
        store.similarity_search("health check", k=1)
        return store
    except Exception as e:
        logger.warning("Existing Chroma_db is unusable (%s). Rebuilding it.", e)
        try:
            if store is None:
                raise RuntimeError("index could not be opened")
            store.delete_collection()
        except Exception:
            shutil.rmtree(PERSIST_DIR, ignore_errors=True)
        return None


# Reuse the existing index to avoid re-indexing on every restart
vectorstores = _open_existing_index()
if vectorstores is None:
    loader = PyPDFLoader(PDF_FILE)
    docs = loader.load()
    for doc in docs:
        doc.page_content = " ".join(doc.page_content.split())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)

    vectorstores = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=PERSIST_DIR
    )

retriever = vectorstores.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# Using groq model for compatibility
model = ChatGroq(model="openai/gpt-oss-120b")

# ==========================================
# 2. FASTAPI BACKEND ENDPOINTS
# ==========================================

app = FastAPI(title="ML Knowledge Assistant")

class QueryRequest(BaseModel):
    query: str

@app.post("/api/query")
def process_query(payload: QueryRequest):
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    # Retrieve context chunks
    try:
        result = retriever.invoke(query)
    except Exception as e:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {type(e).__name__}: {str(e)[:300]}")
    if not result:
        return {
            "found": False,
            "answer": "No relevant context found in the Machine Learning document.",
            "sources": [],
            "latency": 0.0,
            "tokens": None
        }

    context = "\n\n".join([r.page_content for r in result])
    prompt = f"""
You are an AI assistant.

Answer the question only using the context below.

Context:
{context}

Question:
{query}
"""

    start = time.time()
    try:
        response = model.invoke([HumanMessage(content=prompt)])
    except Exception as e:
        logger.exception("LLM call failed")
        raise HTTPException(status_code=502, detail=f"The language model request failed: {type(e).__name__}: {str(e)[:300]}")
    end = time.time()
    latency = round(end - start, 2)

    # Convert response content safely to string
    if isinstance(response.content, str):
        answer_text = response.content
    elif isinstance(response.content, list):
        answer_text = "".join([str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in response.content])
    else:
        answer_text = str(response.content)

    # Extract Token Usage Telemetry
    token_usage = None
    try:
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            token_usage = dict(response.usage_metadata)
    except Exception:
        token_usage = None

    sources = [{"content": doc.page_content, "metadata": doc.metadata} for doc in result]

    return {
        "found": True,
        "answer": answer_text,
        "sources": sources,
        "latency": latency,
        "tokens": token_usage
    }

# ==========================================
# 2b. DOCUMENT MANAGEMENT (additive; /api/query is unchanged)
# ==========================================

UPLOAD_DIR = "uploads"
MAX_UPLOAD_MB = 25
LOADERS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": lambda p: TextLoader(p, encoding="utf-8"),
}
upload_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


def list_documents():
    metas = vectorstores.get(include=["metadatas"]).get("metadatas") or []
    names = {os.path.basename(str(m.get("source", ""))) for m in metas if m}
    return sorted(n for n in names if n)


@app.get("/api/documents")
async def get_documents():
    return {"documents": list_documents()}


def _remove_file(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _purge_from_index(name):
    data = vectorstores.get(include=["metadatas"])
    ids = [i for i, m in zip(data.get("ids", []), data.get("metadatas", []))
           if m and os.path.basename(str(m.get("source", ""))) == name]
    if ids:
        vectorstores.delete(ids=ids)
    return len(ids)


@app.get("/api/health")
def health():
    return {"status": "ok", "documents": len(list_documents())}


@app.post("/api/upload")
def upload_document(file: UploadFile = File(...)):
    name = os.path.basename(file.filename or "")
    ext = os.path.splitext(name)[1].lower()
    if ext not in LOADERS:
        raise HTTPException(status_code=400, detail="Only PDF, DOCX and TXT files are supported.")
    if name in list_documents():
        raise HTTPException(status_code=409, detail=f"'{name}' is already indexed. Remove it first to upload a new version.")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    path = os.path.join(UPLOAD_DIR, name)
    limit = MAX_UPLOAD_MB * 1024 * 1024
    size = 0
    try:
        with open(path, "wb") as out:
            while True:
                block = file.file.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                if size > limit:
                    raise HTTPException(status_code=413, detail=f"File is larger than {MAX_UPLOAD_MB} MB.")
                out.write(block)
        if size == 0:
            raise HTTPException(status_code=400, detail="The file is empty.")

        pages = LOADERS[ext](path).load()
        for p in pages:
            p.page_content = " ".join(p.page_content.split())
            p.metadata["source"] = name
        chunks = upload_splitter.split_documents([p for p in pages if p.page_content])
        if not chunks:
            raise ValueError("no readable text found (scanned or image-only files need OCR first)")
        for i in range(0, len(chunks), 64):
            vectorstores.add_documents(chunks[i:i + 64])
    except HTTPException:
        _remove_file(path)
        raise
    except ImportError as e:
        _remove_file(path)
        logger.exception("Missing package for upload")
        raise HTTPException(status_code=500, detail=f"A package needed for {ext} files is missing on the server: {e}")
    except Exception as e:
        logger.exception("Upload failed")
        _remove_file(path)
        try:
            _purge_from_index(name)
        except Exception:
            pass
        raise HTTPException(status_code=422, detail=f"Could not process '{name}': {str(e)[:300]}")

    return {"name": name, "chunks": len(chunks), "status": "ready"}


@app.delete("/api/documents/{name}")
def delete_document(name: str):
    data = vectorstores.get(include=["metadatas"])
    ids = [i for i, m in zip(data.get("ids", []), data.get("metadatas", []))
           if m and os.path.basename(str(m.get("source", ""))) == name]
    if not ids:
        raise HTTPException(status_code=404, detail=f"'{name}' is not indexed.")
    vectorstores.delete(ids=ids)
    path = os.path.join(UPLOAD_DIR, name)
    if os.path.exists(path):
        os.remove(path)
    return {"name": name, "removed_chunks": len(ids)}


# ==========================================
# 2c. VOICE (additive): speech-to-text via Groq Whisper
# ==========================================

@app.post("/api/transcribe")
def transcribe(file: UploadFile = File(...)):
    audio = file.file.read()
    if not audio:
        raise HTTPException(status_code=400, detail="Empty audio file.")
    try:
        r = groq_client.audio.transcriptions.create(
            file=(file.filename or "audio.webm", audio),
            model="whisper-large-v3-turbo",
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Transcription failed: {e}")
    return {"text": r.text}


# ==========================================
# 3. WEB INTERFACE
# ==========================================

INDEX_HTML = r'''
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
<title>Marginalia | ML Knowledge Assistant</title>
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500&family=Instrument+Serif&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@3.1.6/dist/purify.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@highlightjs/cdn-assets@11.9.0/highlight.min.js"></script>
<style>
:root{--bg:#070a13;--ink:#e9ecf5;--mute:#8d96ad;--glass:rgba(255,255,255,.045);--glass2:rgba(255,255,255,.08);--line:rgba(255,255,255,.09);--a1:#7c8cff;--a2:#38d0c0;--mark:#f5c451;--markbg:rgba(245,196,81,.16);--ok:#3fd08f;--err:#ff7a70;--code:rgba(0,0,0,.35);--sh:0 10px 40px rgba(0,0,0,.35);--o1:#4f46e5;--o2:#0ea5a4;--o3:#7c3aed;color-scheme:dark}
[data-theme=light]{--bg:#eff2f9;--ink:#141a2b;--mute:#5d6680;--glass:rgba(255,255,255,.62);--glass2:rgba(255,255,255,.9);--line:rgba(20,30,70,.11);--a1:#4f46e5;--a2:#0d9488;--mark:#a56a00;--markbg:rgba(245,170,30,.22);--ok:#12805c;--err:#c0352b;--code:rgba(20,30,70,.06);--sh:0 10px 40px rgba(40,50,100,.13);--o1:#a5b4fc;--o2:#99f6e4;--o3:#c4b5fd;color-scheme:light}
*{box-sizing:border-box;margin:0}
html,body{height:100%}
body{font:15px/1.65 Geist,system-ui,sans-serif;background:var(--bg);color:var(--ink);transition:background-color .4s,color .4s;overflow:hidden}
button{font:inherit;color:inherit;cursor:pointer;background:none;border:0}
:focus-visible{outline:2px solid var(--a1);outline-offset:2px;border-radius:8px}
::-webkit-scrollbar{width:8px;height:8px}::-webkit-scrollbar-thumb{background:var(--line);border-radius:8px}::-webkit-scrollbar-thumb:hover{background:var(--mute)}
.amb{position:fixed;inset:0;z-index:0;overflow:hidden;pointer-events:none}
.amb i{position:absolute;width:46vmax;height:46vmax;border-radius:50%;filter:blur(90px);opacity:.28;animation:drift 26s ease-in-out infinite alternate}
.amb i:nth-child(1){background:var(--o1);top:-18%;left:-8%}
.amb i:nth-child(2){background:var(--o2);bottom:-25%;right:-10%;animation-delay:-9s}
.amb i:nth-child(3){background:var(--o3);top:30%;left:45%;width:30vmax;height:30vmax;opacity:.16;animation-delay:-16s}
@keyframes drift{to{transform:translate(8vmax,-5vmax) scale(1.15)}}
.app{position:relative;z-index:1;display:grid;grid-template-columns:316px 1fr;height:100vh;height:100dvh;gap:14px;padding:14px}
.glass{background:var(--glass);border:1px solid var(--line);backdrop-filter:blur(22px) saturate(1.3);-webkit-backdrop-filter:blur(22px) saturate(1.3);border-radius:20px;box-shadow:var(--sh)}
aside{display:flex;flex-direction:column;gap:14px;padding:16px;min-height:0;transition:transform .35s cubic-bezier(.2,.8,.2,1)}
.brand{display:flex;gap:11px;align-items:center}
.logo{width:36px;height:36px;border-radius:11px;background:linear-gradient(135deg,var(--a1),var(--a2));display:grid;place-items:center;color:#fff;font:italic 22px 'Instrument Serif',serif;box-shadow:0 0 22px rgba(124,140,255,.45)}
.brand b{font:400 22px/1 'Instrument Serif',serif;display:block}.brand small{color:var(--mute);font-size:12px}
.btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;border:1px solid var(--line);background:var(--glass2);border-radius:12px;padding:8px 14px;font-weight:500;transition:transform .15s,box-shadow .2s,border-color .2s}
.btn:hover{border-color:var(--a1);transform:translateY(-1px)}.btn:active{transform:scale(.97)}
.btn.pri{background:linear-gradient(135deg,var(--a1),var(--a2));color:#fff;border:0;box-shadow:0 4px 18px rgba(124,140,255,.35)}
.btn.pri:hover{box-shadow:0 6px 26px rgba(124,140,255,.55)}
.btn:disabled{opacity:.45;cursor:not-allowed;transform:none}
.rb{width:40px;height:40px;border-radius:50%;display:grid;place-items:center;flex:none;border:1px solid var(--line);background:var(--glass2);transition:transform .15s,border-color .2s,box-shadow .2s}
.rb svg{width:18px;height:18px;stroke:currentColor;fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
.rb:hover{border-color:var(--a1);transform:scale(1.07)}.rb:active{transform:scale(.93)}
.rb.send,.rb.ok{background:linear-gradient(135deg,var(--a1),var(--a2));color:#fff;border:0;box-shadow:0 4px 18px rgba(124,140,255,.35)}
.rb:disabled{opacity:.45;cursor:not-allowed;transform:none}
.rb.busy svg{display:none}.rb.busy::after{content:"";width:16px;height:16px;border:2px solid var(--line);border-top-color:var(--a1);border-radius:50%;animation:spin .8s linear infinite}
.recbar{flex:1;display:flex;align-items:center;gap:12px;min-width:0}.recbar[hidden]{display:none}
#wave{flex:1;min-width:0;width:100%;height:32px}
#recTime{font:13px 'Geist Mono',monospace;color:var(--mute);min-width:36px;text-align:right}
.rdot{width:8px;height:8px;border-radius:50%;background:var(--err);animation:blink 1.1s ease-in-out infinite;flex:none}
@keyframes blink{50%{opacity:.25}}
.box.rec textarea,.box.rec #mic,.box.rec #send,.box.rec #attach{display:none}
.tabs{display:grid;grid-template-columns:1fr 1fr;background:var(--glass);border:1px solid var(--line);border-radius:12px;padding:3px}
.tabs button{padding:6px;border-radius:9px;color:var(--mute);font-weight:500;font-size:13.5px;transition:.2s}
.tabs button[aria-selected=true]{background:var(--glass2);color:var(--ink);box-shadow:0 1px 6px rgba(0,0,0,.2)}
.pane{flex:1;min-height:0;overflow-y:auto;display:grid;gap:6px;align-content:start}
.pane[hidden]{display:none}
.ci{display:flex;align-items:center;border-radius:11px;transition:background .2s;position:relative}
.ci:hover,.ci.on{background:var(--glass2)}
.ci.on::before{content:"";position:absolute;left:0;top:25%;bottom:25%;width:3px;border-radius:3px;background:linear-gradient(var(--a1),var(--a2))}
.ci .t{flex:1;min-width:0;text-align:left;padding:8px 12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:14px}
.ci input{flex:1;min-width:0;margin:4px 6px;padding:4px 8px;border-radius:8px;border:1px solid var(--a1);background:var(--bg);color:var(--ink);font:inherit}
.ci .ac{display:flex;opacity:0;transition:.2s}.ci:hover .ac,.ci:focus-within .ac{opacity:1}
.ic{padding:6px 8px;color:var(--mute);border-radius:8px;font-size:12.5px}.ic:hover{color:var(--ink);background:var(--glass2)}
.none{color:var(--mute);font-size:13px;padding:10px 12px}
.drop{display:block;border:1.5px dashed var(--line);border-radius:16px;padding:18px 12px;text-align:center;color:var(--mute);font-size:13px;cursor:pointer;transition:.25s}
.drop b{display:block;color:var(--ink);font-weight:500;margin-bottom:2px}
.drop:hover,.drop.over{border-color:var(--a1);background:var(--glass2);box-shadow:0 0 0 4px rgba(124,140,255,.12)}
.dc{display:grid;grid-template-columns:38px 1fr auto;gap:2px 10px;align-items:center;padding:10px;border:1px solid var(--line);border-radius:14px;background:var(--glass);animation:in .35s both}
.fi{grid-row:span 2;width:38px;height:38px;border-radius:10px;display:grid;place-items:center;font:600 10.5px 'Geist Mono',monospace;color:#fff}
.fi.pdf{background:linear-gradient(135deg,#ef5350,#c62828)}.fi.docx{background:linear-gradient(135deg,#5c8dff,#2745c4)}.fi.txt{background:linear-gradient(135deg,#7a8599,#4a5468)}
.dn{font-size:13.5px;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0}
.dm{font-size:12px;color:var(--mute);grid-column:2/4;display:flex;gap:8px;align-items:center}
.st{font-size:11.5px;border-radius:99px;padding:0 8px;border:1px solid var(--line)}.st.ready{color:var(--ok)}.st.error{color:var(--err)}
.bar{height:4px;flex:1;border-radius:4px;background:var(--line);overflow:hidden}.bar i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--a1),var(--a2));transition:width .2s}
.sk{background:linear-gradient(90deg,var(--line) 25%,var(--glass2) 50%,var(--line) 75%);background-size:200% 100%;animation:sh 1.3s linear infinite;border-radius:6px;height:9px;grid-column:2/4}
@keyframes sh{to{background-position:-200% 0}}
main{display:flex;flex-direction:column;min-width:0;min-height:0;overflow:hidden}
.top{display:flex;align-items:center;gap:10px;padding:12px 16px;border-bottom:1px solid var(--line)}
.top h2{flex:1;font-size:14px;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#menu{display:none}
.tg{position:relative;width:52px;height:28px;border-radius:99px;border:1px solid var(--line);background:var(--glass2)}
.tg::after{content:"";position:absolute;top:3px;left:3px;width:20px;height:20px;border-radius:50%;background:linear-gradient(135deg,var(--a1),var(--a2));transition:transform .35s cubic-bezier(.3,1.4,.5,1)}
[data-theme=light] .tg::after{transform:translateX(24px);background:linear-gradient(135deg,#ffc94d,#ff9f43)}
#scroll{flex:1;overflow-y:auto;padding:28px 20px 8px;scroll-behavior:smooth}
.col{max-width:800px;margin:0 auto;display:grid;gap:26px}
.hero{padding:7vh 0 4px}
.hero h1{font:400 clamp(34px,5.4vw,56px)/1.05 'Instrument Serif',serif;letter-spacing:-.01em;max-width:15ch}
.hero p{color:var(--mute);max-width:54ch;margin:14px 0 26px}
.sug{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px}
.sug button{text-align:left;padding:14px 16px;border-radius:16px;border:1px solid var(--line);background:var(--glass);transition:.25s;font-size:14px}
.sug button:hover{transform:translateY(-3px);border-color:var(--a1);box-shadow:0 10px 30px rgba(124,140,255,.18)}
.in{animation:in .4s cubic-bezier(.2,.8,.2,1) both}
@keyframes in{from{opacity:0;transform:translateY(10px)}}
.u{display:flex;justify-content:flex-end}
.u div{background:linear-gradient(135deg,var(--a1),#6573ee);color:#fff;padding:10px 16px;border-radius:20px 20px 5px 20px;max-width:80%;white-space:pre-wrap;word-break:break-word;box-shadow:0 6px 20px rgba(124,140,255,.25)}
.ai{display:flex;gap:14px;align-items:flex-start}
.av{width:32px;height:32px;border-radius:10px;flex:none;background:linear-gradient(135deg,var(--a1),var(--a2));display:grid;place-items:center;color:#fff;font:italic 18px 'Instrument Serif',serif}
.bub{flex:1;min-width:0}
.md>*+*{margin-top:.8em}.md ul,.md ol{padding-left:1.4em}.md li+li{margin-top:.25em}
.md h1,.md h2,.md h3{font:400 22px/1.2 'Instrument Serif',serif;margin-top:1.2em}
.md code{font:13px 'Geist Mono',monospace;background:var(--code);padding:1px 6px;border-radius:6px}
.cb{border:1px solid var(--line);border-radius:14px;overflow:hidden;background:var(--code)}
.ch{display:flex;justify-content:space-between;align-items:center;padding:5px 8px 5px 14px;border-bottom:1px solid var(--line);font:12px 'Geist Mono',monospace;color:var(--mute)}
.ch button{padding:2px 10px;border-radius:7px;font:12px Geist}.ch button:hover{background:var(--glass2);color:var(--ink)}
.cb pre{padding:12px 14px;overflow-x:auto}.cb code{background:none;padding:0}
.hljs-keyword,.hljs-built_in{color:#c792ea}.hljs-string{color:#7fd88f}.hljs-number,.hljs-literal{color:#f9a86a}.hljs-comment{color:var(--mute);font-style:italic}.hljs-title,.hljs-function{color:#6cb6ff}
.md table{border-collapse:collapse;display:block;overflow-x:auto}.md td,.md th{border:1px solid var(--line);padding:6px 12px}
.acts{display:flex;flex-wrap:wrap;gap:4px;margin-top:10px;color:var(--mute);font-size:12.5px;align-items:center}
.acts .ic.on{color:var(--a2)}.acts .sp{flex:1}
.src{margin-top:16px;padding-top:14px;border-top:1px solid var(--line)}
.src h4{font-size:12.5px;font-weight:500;color:var(--mute);margin-bottom:10px}
.sc{border:1px solid var(--line);border-left:3px solid var(--mark);border-radius:12px;background:var(--glass);margin-bottom:8px;transition:.2s}
.sc:hover{background:var(--glass2)}
.sh{display:flex;gap:10px;align-items:center;width:100%;padding:9px 12px;text-align:left;font-size:13.5px}
.sh .nm{flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sh .n{width:20px;height:20px;border-radius:6px;background:var(--markbg);color:var(--mark);display:grid;place-items:center;font:500 11.5px 'Geist Mono',monospace;flex:none}
.pg{font:11.5px 'Geist Mono',monospace;color:var(--mute);border:1px solid var(--line);border-radius:6px;padding:0 7px}
.sb{display:none;padding:0 12px 12px;font-size:13px;color:var(--mute)}.sc.open .sb{display:block}
.sb p{max-height:150px;overflow:auto;white-space:pre-wrap;margin-bottom:8px}
mark{background:var(--markbg);color:var(--mark);border-radius:3px;padding:0 2px}
.meta{font:12px 'Geist Mono',monospace;color:var(--mute)}
.think{display:flex;gap:12px;align-items:center;color:var(--mute);padding:6px 0}
.orb{width:32px;height:32px;border-radius:10px;background:conic-gradient(var(--a1),var(--a2),var(--o3),var(--a1));animation:spin 2.2s linear infinite;box-shadow:0 0 24px rgba(124,140,255,.5);flex:none}
@keyframes spin{to{transform:rotate(360deg)}}
.shim{background:linear-gradient(90deg,var(--mute),var(--ink),var(--mute));background-size:200% 100%;-webkit-background-clip:text;background-clip:text;color:transparent;animation:sh 1.8s linear infinite}
.err{color:var(--err)}
.comp{padding:8px 20px calc(16px + env(safe-area-inset-bottom))}
.box{max-width:800px;margin:0 auto;display:flex;gap:8px;align-items:flex-end;padding:8px 8px 8px 18px;border-radius:22px;transition:.25s}
.box:focus-within{border-color:var(--a1);box-shadow:0 0 0 4px rgba(124,140,255,.15),var(--sh)}
textarea{flex:1;border:0;outline:0;background:none;color:var(--ink);font:inherit;resize:none;max-height:180px;padding:8px 0}
.hint{max-width:800px;margin:8px auto 0;text-align:center;font-size:12px;color:var(--mute)}
.toast{position:fixed;left:50%;bottom:92px;transform:translate(-50%,20px);background:var(--glass2);backdrop-filter:blur(18px);border:1px solid var(--line);padding:9px 16px;border-radius:12px;opacity:0;pointer-events:none;transition:.3s;z-index:60;box-shadow:var(--sh);font-size:13.5px}
.toast.on{opacity:1;transform:translate(-50%,0)}
.modal{position:fixed;inset:0;z-index:50;display:none;place-items:center;background:rgba(3,6,15,.6);backdrop-filter:blur(6px);padding:16px}
.modal.on{display:grid;animation:in .25s both}
.mc{width:min(720px,100%);max-height:86vh;display:flex;flex-direction:column;padding:20px;background:var(--bg)}
.mc header{display:flex;gap:10px;align-items:center;margin-bottom:12px}.mc header b{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mc .body{overflow:auto;white-space:pre-wrap;color:var(--mute);line-height:1.75}
.scrim{display:none}
@media(max-width:900px){
 .app{grid-template-columns:1fr;padding:0;gap:0}
 main{border-radius:0;border:0}
 aside{position:fixed;z-index:30;inset:0 auto 0 0;width:min(86vw,340px);border-radius:0 22px 22px 0;transform:translateX(-105%);background:var(--bg)}
 body.open aside{transform:none}
 body.open .scrim{display:block;position:fixed;inset:0;z-index:25;background:rgba(3,6,15,.55);animation:in .3s both}
 #menu{display:inline-flex}.hint{display:none}.ci .ac{opacity:1}
 .btn,.ic,.tabs button{min-height:40px}
}
@media(prefers-reduced-motion:reduce){*,::before,::after{animation-duration:.01ms!important;animation-iteration-count:1!important;transition:none!important}}
</style>
</head>
<body>
<div class="amb" aria-hidden="true"><i></i><i></i><i></i></div>
<div class="scrim" id="scrim"></div>
<div class="app">
<aside class="glass" id="side" aria-label="Sidebar">
  <div class="brand"><div class="logo">M</div><div><b>Marginalia</b><small>ML document intelligence</small></div></div>
  <button class="btn pri" id="newChat">New conversation</button>
  <div class="tabs" role="tablist"><button role="tab" id="tabC" aria-selected="true">Conversations</button><button role="tab" id="tabD" aria-selected="false">Documents</button></div>
  <div class="pane" id="chats" role="tabpanel"></div>
  <div class="pane" id="docsPane" role="tabpanel" hidden>
    <label class="drop" id="drop"><b>Drop files to add knowledge</b>PDF, DOCX or TXT, up to 25 MB<input type="file" id="file" accept=".pdf,.docx,.txt" multiple hidden></label>
    <div id="docs" style="display:grid;gap:8px"></div>
  </div>
</aside>
<main class="glass">
  <header class="top">
    <button class="btn" id="menu" aria-label="Open menu">Menu</button>
    <h2 id="title">New conversation</h2>
    <button class="tg" id="theme" role="switch" aria-label="Toggle light mode"></button>
  </header>
  <div id="scroll"><div class="col" id="thread" aria-live="polite"></div></div>
  <div class="comp">
    <div class="box glass">
      <button class="rb" id="attach" aria-label="Upload documents"><svg viewBox="0 0 24 24"><path d="M21 12.5l-8.6 8.6a5.5 5.5 0 0 1-7.8-7.8l9-9a3.7 3.7 0 0 1 5.2 5.2l-9 9a1.8 1.8 0 0 1-2.6-2.6l8.4-8.4"/></svg></button>
      <textarea id="q" rows="1" placeholder="Ask about your documents" aria-label="Question"></textarea>
      <div class="recbar" id="recbar" hidden>
        <button class="rb" id="recCancel" aria-label="Cancel recording"><svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18"/></svg></button>
        <span class="rdot" aria-hidden="true"></span><canvas id="wave" aria-hidden="true"></canvas><span id="recTime" aria-live="off">0:00</span>
        <button class="rb ok" id="recDone" aria-label="Finish and transcribe"><svg viewBox="0 0 24 24"><path d="M5 12l5 5 9-10"/></svg></button>
      </div>
      <button class="rb" id="mic" aria-label="Start voice input"><svg viewBox="0 0 24 24"><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/></svg></button>
      <button class="rb send" id="send" aria-label="Send message"><svg viewBox="0 0 24 24"><path d="M12 19V5M6 11l6-6 6 6"/></svg></button>
    </div>
    <div class="hint">Enter to send, Shift + Enter for a new line. Answers come only from your documents.</div>
  </div>
</main>
</div>
<div class="modal" id="modal" role="dialog" aria-modal="true" aria-label="Source preview"><div class="mc glass"><header><span class="pg" id="mpg"></span><b id="mnm"></b><button class="btn" id="mx">Close</button></header><div class="body" id="mbody"></div></div></div>
<div class="toast" id="toast" role="status"></div>
<script>
const $=(s,r=document)=>r.querySelector(s);
const el=(t,c,x)=>{const e=document.createElement(t);if(c)e.className=c;if(x!==undefined)e.textContent=x;return e};
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const KEY='rag.chats.v1',reduce=matchMedia('(prefers-reduced-motion: reduce)').matches;
let chats=[],cur=null,busy=false,docs=[],tab='c';
try{chats=JSON.parse(localStorage.getItem(KEY)||'[]')}catch(e){}
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(chats.slice(0,50)))}catch(e){}};
const active=()=>chats.find(c=>c.id===cur);
const SUG=['Summarize the main topics covered in my documents','What is the difference between supervised and unsupervised learning?','Explain overfitting and show a short Python example of regularization','Which evaluation metrics are used for classification?'];
function toast(m){const t=$('#toast');t.textContent=m;t.classList.add('on');clearTimeout(toast.t);toast.t=setTimeout(()=>t.classList.remove('on'),2400)}
const copy=async(t,msg='Copied')=>{try{await navigator.clipboard.writeText(t);toast(msg)}catch(e){toast('Copy failed')}};

/* theme */
function setTheme(t,anim){
  const go=()=>{document.documentElement.dataset.theme=t;$('#theme').setAttribute('aria-checked',t==='light')};
  try{localStorage.setItem('rag.theme',t)}catch(e){}
  if(anim&&document.startViewTransition&&!reduce)document.startViewTransition(go);else go();
}
let th0='dark';try{th0=localStorage.getItem('rag.theme')||'dark'}catch(e){}
setTheme(th0,false);
$('#theme').onclick=()=>setTheme(document.documentElement.dataset.theme==='dark'?'light':'dark',true);

/* drawer + tabs */
const closeSide=()=>document.body.classList.remove('open');
$('#menu').onclick=()=>document.body.classList.add('open');$('#scrim').onclick=closeSide;
function setTab(t){tab=t;$('#tabC').setAttribute('aria-selected',t==='c');$('#tabD').setAttribute('aria-selected',t==='d');$('#chats').hidden=t!=='c';$('#docsPane').hidden=t!=='d'}
$('#tabC').onclick=()=>setTab('c');$('#tabD').onclick=()=>setTab('d');

/* markdown */
marked.use({renderer:{code(code,lang){
  const l=lang&&hljs.getLanguage(lang)?lang:'plaintext';
  return `<div class="cb"><div class="ch"><span>${esc(l)}</span><button data-copy>Copy</button></div><pre><code class="hljs">${hljs.highlight(code,{language:l}).value}</code></pre></div>`}}});
document.addEventListener('click',e=>{const b=e.target.closest('[data-copy]');if(b)copy(b.closest('.cb').querySelector('code').innerText,'Code copied')});

/* chats */
function newChat(){
  const e=chats.find(c=>!c.messages.length);
  if(e)cur=e.id;else{const c={id:'c'+Date.now(),title:'New conversation',messages:[]};chats.unshift(c);cur=c.id}
  renderAll();closeSide();setTab('c');$('#q').focus();
}
$('#newChat').onclick=newChat;
function chatRow(c){
  const row=el('div','ci'+(c.id===cur?' on':''));
  const b=el('button','t',c.title);b.title=c.title;b.onclick=()=>{cur=c.id;renderAll();closeSide()};
  const ac=el('div','ac');
  const rn=el('button','ic','Rename');rn.setAttribute('aria-label','Rename conversation');
  rn.onclick=()=>{
    const i=el('input');i.value=c.title;i.setAttribute('aria-label','Conversation name');b.replaceWith(i);ac.remove();i.focus();i.select();
    const done=ok=>{if(ok&&i.value.trim()){c.title=i.value.trim();save()}renderAll()};
    i.onkeydown=e=>{if(e.key==='Enter')done(true);if(e.key==='Escape')done(false)};i.onblur=()=>done(true);
  };
  const dl=el('button','ic','Delete');dl.setAttribute('aria-label','Delete conversation');
  dl.onclick=()=>{chats=chats.filter(k=>k.id!==c.id);if(cur===c.id)cur=null;save();if(!cur)newChat();else renderAll();toast('Conversation deleted')};
  ac.append(rn,dl);row.append(b,ac);return row;
}
function renderChats(){
  const box=$('#chats');box.innerHTML='';
  const l=chats.filter(c=>c.messages.length);
  if(!l.length)box.appendChild(el('div','none','Your conversations will appear here.'));
  l.forEach(c=>box.appendChild(chatRow(c)));
}
function renderThread(anim){
  const c=active(),thd=$('#thread');thd.innerHTML='';
  $('#title').textContent=c&&c.messages.length?c.title:'New conversation';
  if(!c||!c.messages.length){
    const h=el('section','hero in');
    h.append(el('h1','','Ask your documents. See the evidence.'),el('p','','Every answer is written from your own files and links back to the exact passages it used.'));
    const s=el('div','sug');
    SUG.forEach(q=>{const b=el('button','',q);b.onclick=()=>{$('#q').value=q;ask()};s.appendChild(b)});
    h.appendChild(s);thd.appendChild(h);return;
  }
  c.messages.forEach((m,i)=>{
    const r=m.role==='user'?userRow(m.text):aiRow(m,c,i);
    if(anim&&i>=c.messages.length-1)r.classList.add('in');thd.appendChild(r);
  });
}
function renderAll(anim){renderChats();renderThread(anim);if(!anim)toEnd(false)}
const toEnd=(smooth=true)=>requestAnimationFrame(()=>$('#scroll').scrollTo({top:$('#scroll').scrollHeight,behavior:smooth&&!reduce?'smooth':'auto'}));

/* messages */
const userRow=t=>{const r=el('div','u');r.appendChild(el('div','',t));return r};
function shell(){const r=el('div','ai'),b=el('div','bub');r.append(el('div','av','M'),b);return[r,b]}
const STEPS=['Searching your documents','Reading the best passages','Writing the answer'];
function thinkRow(){
  const[r,b]=shell();r.classList.add('in');r.firstChild.remove();
  const t=el('div','think');const s=el('span','shim',STEPS[0]);t.append(el('div','orb'),s);b.appendChild(t);
  let i=0;r._t=setInterval(()=>{i=Math.min(i+1,STEPS.length-1);s.textContent=STEPS[i]},1600);return r;
}
const STOP=new Set(['what','that','this','with','from','about','does','explain','difference','between','which','their','there','where','when','show','short','example']);
function hl(text,q){
  const terms=[...new Set((q.toLowerCase().match(/[a-z0-9]{4,}/g)||[]).filter(w=>!STOP.has(w)))];
  const safe=esc(text);if(!terms.length)return safe;
  return safe.replace(new RegExp('\\b('+terms.join('|')+')','gi'),'<mark>$1</mark>');
}
function openModal(name,pg,text,q){$('#mnm').textContent=name;$('#mpg').textContent=pg||'Excerpt';$('#mbody').innerHTML=hl(text,q);$('#modal').classList.add('on');$('#mx').focus()}
const closeModal=()=>$('#modal').classList.remove('on');
$('#mx').onclick=closeModal;$('#modal').onclick=e=>{if(e.target.id==='modal')closeModal()};
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeModal()});

function aiRow(m,c,idx){
  const[r,b]=shell();
  if(m.error){b.appendChild(el('div','err',m.text));b.appendChild(actions(m,c,idx,true));return r}
  const body=el('div','md');
  if(!m.found){body.style.color='var(--mute)';body.textContent=m.text+' Try rephrasing with terms from your documents.';b.appendChild(body);return r}
  body.innerHTML=DOMPurify.sanitize(marked.parse(m.text));b.appendChild(body);b.appendChild(actions(m,c,idx));
  const q=(c.messages[idx-1]||{}).text||'',src=m.sources||[];
  if(src.length){
    const s=el('section','src');s.appendChild(el('h4','',`Sources \u00b7 ${src.length} passages retrieved from your documents`));
    src.forEach((x,i)=>{
      const md=x.metadata||{},name=String(md.source||'Document').split(/[\\/]/).pop(),pg=md.page!==undefined?'p.'+(md.page+1):'';
      const card=el('div','sc'),h=el('button','sh');h.setAttribute('aria-expanded','false');
      h.append(el('span','n',i+1),el('span','nm',name),...(pg?[el('span','pg',pg)]:[]));
      const sb=el('div','sb'),p=el('p');p.innerHTML=hl(x.content,q);
      const full=el('button','btn','Open preview');full.onclick=()=>openModal(name,pg,x.content,q);sb.append(p,full);
      h.onclick=()=>{const o=card.classList.toggle('open');h.setAttribute('aria-expanded',o)};
      card.append(h,sb);s.appendChild(card);
    });
    b.appendChild(s);
  }
  return r;
}
function actions(m,c,idx,errOnly){
  const a=el('div','acts');
  if(!errOnly){const cp=el('button','ic','Copy');cp.onclick=()=>copy(m.text);a.appendChild(cp)}
  if(idx===c.messages.length-1){const rg=el('button','ic','Regenerate');rg.onclick=()=>regen(c);a.appendChild(rg)}
  if(!errOnly){
    [['up','Helpful'],['down','Not helpful']].forEach(([v,l])=>{
      const f=el('button','ic'+(m.fb===v?' on':''),l);f.setAttribute('aria-pressed',m.fb===v);
      f.onclick=()=>{m.fb=m.fb===v?null:v;save();renderThread();if(m.fb)toast('Thanks for the feedback')};a.appendChild(f);
    });
    a.appendChild(el('span','sp'));
    const bits=[m.latency+'s'];if(m.tokens&&typeof m.tokens==='object')bits.push((m.tokens.input_tokens??'-')+' in / '+(m.tokens.output_tokens??'-')+' out');
    a.appendChild(el('span','meta',bits.join(' \u00b7 ')));
  }
  return a;
}

/* ask / regenerate */
async function readJson(r){const t=await r.text();try{return JSON.parse(t)}catch(e){return{detail:r.ok?'Unexpected response from the server.':'Server error ('+r.status+'). Check the server logs for details.'}}}
const input=$('#q');
input.addEventListener('input',()=>{input.style.height='auto';input.style.height=input.scrollHeight+'px'});
input.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();ask()}});
$('#send').onclick=ask;
function ask(){
  const query=input.value.trim();if(!query||busy)return;
  if(!active())newChat();const c=active();
  input.value='';input.style.height='auto';
  if(!c.messages.length)c.title=query.length>44?query.slice(0,44)+'...':query;
  c.messages.push({role:'user',text:query});save();renderAll(true);run(c,query);
}
function regen(c){
  if(busy)return;const last=c.messages[c.messages.length-1];if(!last||last.role!=='ai')return;
  c.messages.pop();const u=c.messages[c.messages.length-1];renderAll();run(c,u.text);
}
async function run(c,query){
  busy=true;$('#send').disabled=true;
  const tr=thinkRow();$('#thread').appendChild(tr);toEnd();
  try{
    const res=await fetch('/api/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query})});
    const d=await readJson(res);
    if(!res.ok)throw new Error(typeof d.detail==='string'?d.detail:'The request failed.');
    c.messages.push({role:'ai',found:d.found,text:String(d.answer||''),sources:d.sources,latency:d.latency,tokens:d.tokens});
  }catch(err){c.messages.push({role:'ai',error:true,text:'Something went wrong: '+err.message+' Check that the server is running, then try again.'})}
  clearInterval(tr._t);busy=false;$('#send').disabled=false;save();renderAll(true);toEnd();input.focus();
}

/* documents */
const extOf=n=>(n.split('.').pop()||'').toLowerCase();
const fmt=b=>b>1048576?(b/1048576).toFixed(1)+' MB':b>1024?Math.round(b/1024)+' KB':b+' B';
function renderDocs(){
  const box=$('#docs');box.innerHTML='';$('#tabD').textContent='Documents'+(docs.length?' ('+docs.length+')':'');
  if(!docs.length)box.appendChild(el('div','none','No documents yet. Add a file to start asking questions.'));
  docs.forEach(d=>{
    const c=el('div','dc');c.appendChild(el('div','fi '+d.ext,d.ext.toUpperCase().slice(0,4)));
    c.appendChild(el('div','dn',d.name)).title=d.name;
    if(d.status==='ready'||d.status==='error'){
      const x=el('button','ic','Remove');x.setAttribute('aria-label','Remove '+d.name);x.onclick=()=>removeDoc(d);c.appendChild(x);
    }else c.appendChild(el('span'));
    const m=el('div','dm');
    if(d.status==='uploading'){const b=el('div','bar'),i=el('i');i.style.width=d.pct+'%';b.appendChild(i);m.append(el('span','',fmt(d.size||0)),b,el('span','',d.pct+'%'))}
    else if(d.status==='processing'){m.textContent='Indexing passages...';c.append(m,el('div','sk'),Object.assign(el('div','sk'),{style:'width:60%'}));box.appendChild(c);return}
    else{m.append(el('span','',[d.size?fmt(d.size):'',d.ext.toUpperCase()].filter(Boolean).join(' \u00b7 ')),el('span','st '+d.status,d.status==='ready'?'Ready':'Failed'));if(d.detail){c.title=d.detail}}
    c.appendChild(m);if(d.status==='error'&&d.detail){const e=el('div','dm err',d.detail);c.appendChild(e)}
    box.appendChild(c);
  });
}
async function loadDocs(){
  try{
    const r=await fetch('/api/documents');if(!r.ok)return;
    const names=((await r.json()).documents)||[];
    const local=docs.filter(x=>x.status!=='ready'||names.includes(x.name));
    names.forEach(n=>{if(!local.some(x=>x.name===n))local.push({name:n,ext:extOf(n),status:'ready'})});
    docs=local;renderDocs();
  }catch(e){}
}
async function removeDoc(d){
  if(d.status==='error'){docs=docs.filter(x=>x!==d);renderDocs();return}
  try{
    const r=await fetch('/api/documents/'+encodeURIComponent(d.name),{method:'DELETE'});
    if(!r.ok)throw new Error(r.status===404||r.status===405?'Removing documents needs the DELETE /api/documents endpoint on the server.':'Could not remove the document.');
    docs=docs.filter(x=>x!==d);renderDocs();toast('Removed '+d.name);
  }catch(e){toast(e.message)}
}
function upload(files){
  files.forEach(f=>{
    const ext=extOf(f.name),d={name:f.name,ext,size:f.size,status:'uploading',pct:0};
    docs=docs.filter(x=>x.name!==f.name);docs.unshift(d);setTab('d');renderDocs();
    const fail=m=>{d.status='error';d.detail=m;renderDocs()};
    if(!['pdf','docx','txt'].includes(ext))return fail('Only PDF, DOCX and TXT files are supported.');
    if(f.size>25*1048576)return fail('File is larger than 25 MB.');
    const fd=new FormData();fd.append('file',f);const x=new XMLHttpRequest();
    x.open('POST','/api/upload');
    x.upload.onprogress=e=>{if(e.lengthComputable){d.pct=Math.round(e.loaded/e.total*100);renderDocs()}};
    x.upload.onload=()=>{d.status='processing';renderDocs()};
    x.onload=()=>{let j={};try{j=JSON.parse(x.responseText)}catch(e){}
      if(x.status>=200&&x.status<300){d.status='ready';d.detail=j.chunks+' passages indexed';renderDocs();toast(f.name+' is ready')}
      else fail(typeof j.detail==='string'?j.detail:'Upload failed (server error '+x.status+'). Check the server logs.')};
    x.onerror=()=>fail('Connection lost. The server may have restarted (out of memory) or is unreachable.');
    x.send(fd);
  });
}
$('#attach').onclick=()=>$('#file').click();
$('#file').onchange=e=>{upload([...e.target.files]);e.target.value=''};
const dz=$('#drop');
['dragover','dragenter'].forEach(v=>dz.addEventListener(v,e=>{e.preventDefault();dz.classList.add('over')}));
['dragleave','drop'].forEach(v=>dz.addEventListener(v,e=>{e.preventDefault();dz.classList.remove('over')}));
dz.addEventListener('drop',e=>upload([...e.dataTransfer.files]));
window.addEventListener('dragover',e=>e.preventDefault());window.addEventListener('drop',e=>e.preventDefault());

/* voice input (Groq Whisper via /api/transcribe) */
const mic=$('#mic'),vbox=$('.box'),vbar=$('#recbar'),cv=$('#wave'),vtime=$('#recTime');
let rec=null,chunks=[],vstream=null,actx=null,raf=0,tick=0,t0=0,cancelled=false,levels=[];
if(!navigator.mediaDevices||!window.MediaRecorder)mic.style.display='none';
function stopMeter(){
  cancelAnimationFrame(raf);clearInterval(tick);
  if(actx){actx.close();actx=null}
  if(vstream){vstream.getTracks().forEach(t=>t.stop());vstream=null}
  vbox.classList.remove('rec');vbar.hidden=true;
}
async function startRec(){
  try{vstream=await navigator.mediaDevices.getUserMedia({audio:true})}
  catch(e){toast('Microphone access was blocked.');return}
  rec=new MediaRecorder(vstream);chunks=[];cancelled=false;levels=[];
  rec.ondataavailable=e=>{if(e.data.size)chunks.push(e.data)};
  rec.onstop=onStop;
  actx=new(window.AudioContext||window.webkitAudioContext)();
  const an=actx.createAnalyser();an.fftSize=256;actx.createMediaStreamSource(vstream).connect(an);
  const data=new Uint8Array(an.fftSize);
  vbox.classList.add('rec');vbar.hidden=false;
  const dpr=devicePixelRatio||1;cv.width=cv.clientWidth*dpr;cv.height=32*dpr;
  const ctx=cv.getContext('2d'),color=getComputedStyle(document.documentElement).getPropertyValue('--a1').trim();
  const bw=3*dpr,gap=3*dpr;let f=0;
  const loop=()=>{
    an.getByteTimeDomainData(data);
    if(++f%3===0){let s=0;for(const v of data)s+=Math.abs(v-128);levels.push(Math.min(1,s/data.length/28));
      const max=Math.floor(cv.width/(bw+gap));if(levels.length>max)levels.shift()}
    ctx.clearRect(0,0,cv.width,cv.height);ctx.fillStyle=color;
    levels.forEach((l,i)=>{const h=Math.max(3*dpr,l*cv.height),x=cv.width-(levels.length-i)*(bw+gap);
      ctx.beginPath();ctx.roundRect(x,(cv.height-h)/2,bw,h,bw/2);ctx.fill()});
    raf=requestAnimationFrame(loop);
  };
  t0=Date.now();vtime.textContent='0:00';
  tick=setInterval(()=>{const s=Math.floor((Date.now()-t0)/1000);vtime.textContent=Math.floor(s/60)+':'+String(s%60).padStart(2,'0');if(s>=120)finishRec()},250);
  rec.start();loop();
}
function finishRec(){cancelled=false;if(rec&&rec.state==='recording')rec.stop()}
function cancelRec(){cancelled=true;if(rec&&rec.state==='recording')rec.stop()}
async function onStop(){
  const type=rec.mimeType||'audio/webm',ext=type.includes('mp4')?'mp4':type.includes('ogg')?'ogg':'webm',short=Date.now()-t0<700;
  stopMeter();
  if(cancelled)return;
  if(short||!chunks.length){toast('Recording was too short. Hold on a little longer.');return}
  mic.classList.add('busy');mic.disabled=true;
  try{
    const fd=new FormData();fd.append('file',new Blob(chunks,{type}),'audio.'+ext);
    const r=await fetch('/api/transcribe',{method:'POST',body:fd});
    const d=await readJson(r);
    if(!r.ok)throw new Error(typeof d.detail==='string'?d.detail:'Transcription failed.');
    const text=(d.text||'').trim();
    if(!text)toast('No speech detected.');
    else{input.value=(input.value?input.value+' ':'')+text;input.dispatchEvent(new Event('input'))}
    input.focus();
  }catch(e){toast(e.message)}
  mic.classList.remove('busy');mic.disabled=false;
}
mic.onclick=startRec;$('#recDone').onclick=finishRec;$('#recCancel').onclick=cancelRec;
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&rec&&rec.state==='recording')cancelRec()});

cur=(chats.find(c=>c.messages.length)||{}).id||null;
if(!cur)newChat();else renderAll();
renderDocs();loadDocs();
</script>
</body>
</html>
'''

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return INDEX_HTML