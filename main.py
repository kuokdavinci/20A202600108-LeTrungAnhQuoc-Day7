import os
import sys
from pathlib import Path

# Ensure the terminal handles UTF-8 for printing special characters (Vietnamese, quotes, etc.)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from openai import OpenAI
from src.models import Document
from src.store import EmbeddingStore

# Load all 20 documents for a comprehensive test if they exist
try:
    wiki_path = Path("data/wiki_docs")
    if wiki_path.exists():
        SAMPLE_FILES = sorted([str(f) for f in wiki_path.glob("*.txt")])
    else:
        SAMPLE_FILES = []
except Exception:
    SAMPLE_FILES = []


def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """Load documents from file paths for the manual demo."""
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []
    
    for raw_path in file_paths:
        path = Path(raw_path)

        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path} (allowed: .md, .txt)")
            continue

        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue

        content_lines = path.read_text(encoding="utf-8").splitlines()
        
        # Check if the first line is a URL as per report description
        if content_lines and content_lines[0].startswith("http"):
            source_url = content_lines[0].strip()
            content = "\n".join(content_lines[1:])
        else:
            source_url = "N/A"
            content = "\n".join(content_lines)
            
        # Parse category from filename (e.g. "Gaming_Bullet_Kin" -> category="Gaming")
        stem = path.stem
        category = "General"
        if "_" in stem:
            parts = stem.split("_", 1)
            category = parts[0]
            doc_id = parts[1]
        else:
            doc_id = stem

        documents.append(
            Document(
                id=doc_id,
                content=content,
                metadata={
                    "source": str(path), 
                    "extension": path.suffix.lower(),
                    "category": category,
                    "source_url": source_url
                },
            )
        )

    return documents


# Removed demo_llm mock because we now use src/llms.py


def run_manual_demo(question: str | None = None, sample_files: list[str] | None = None) -> int:
    files = sample_files or SAMPLE_FILES
    query = question or "Summarize the key information from the loaded files."

    print("=== Manual File Test ===")
    print("Accepted file types: .md, .txt")
    print("Input file list:")
    for file_path in files:
        print(f"  - {file_path}")

    docs = load_documents_from_files(files)
    if not docs:
        print("\nNo valid input files were loaded.")
        print("Create files matching the sample paths above, then rerun:")
        print("  python3 main.py")
        return 1

    print(f"\nLoaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.id}: {doc.metadata['source']}")

    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    print(f"\nEmbedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    # 2. Strategy & Store
    strategy_name = os.getenv("CHUNKING_STRATEGY", "recursive").strip().lower()
    print(f"Using chunking strategy: {strategy_name}")
    
    if strategy_name == "semantic":
        from src.chunking import SemanticChunker
        chunker = SemanticChunker(embedding_fn=embedder, threshold=0.5)
    elif strategy_name == "by_sentences":
        from src.chunking import SentenceChunker
        chunker = SentenceChunker(max_sentences_per_chunk=3)
    elif strategy_name == "fixed_size":
        from src.chunking import FixedSizeChunker
        chunker = FixedSizeChunker(chunk_size=500, overlap=50)
    else:
        from src.chunking import RecursiveChunker
        chunker = RecursiveChunker(chunk_size=500)
    
    store = EmbeddingStore(collection_name="manual_test_store", embedding_fn=embedder, chunker=chunker)
    store.add_documents(docs)

    print(f"\nStored {store.get_collection_size()} documents in EmbeddingStore")
    print("\n=== EmbeddingStore Search Test ===")
    print(f"Query: {query}")
    search_results = store.search(query, top_k=3)
    for index, result in enumerate(search_results, start=1):
        metadata = result.get("metadata", {})
        print(f"{index}. score={result['score']:.3f} source={metadata.get('source')}")
        print(f"   category={metadata.get('category', 'General')}")
        print(f"   content preview: {result['content'][:120].replace(chr(10), ' ')}...")

    print("\n=== KnowledgeBaseAgent Test ===")
    
    # Agent now handles LLM internally via Ollama/OpenAI API
    agent = KnowledgeBaseAgent(store=store)
    
    print(f"Question: {query}")
    print("Agent answer:")
    print("-" * 40)
    print(agent.answer(query, top_k=3))
    print("-" * 40)
    return 0


def main() -> int:
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
    return run_manual_demo(question=question)


if __name__ == "__main__":
    raise SystemExit(main())
