import os
from dotenv import load_dotenv
from src.chunking import ChunkingStrategyComparator
from src.embeddings import OpenAIEmbedder, LocalEmbedder, _mock_embed, EMBEDDING_PROVIDER_ENV

def main():
    load_dotenv()
    
    # 0. Initialize Embedder for Semantic Chunking
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "openai":
        embedder = OpenAIEmbedder()
    elif provider == "local":
        try:
            embedder = LocalEmbedder()
        except ImportError:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    from pathlib import Path
    sample_file = Path("data/wiki_docs/Gaming_Bullet_Kin.txt")
    if not sample_file.exists():
        print(f"Error: {sample_file} not found.")
        return

    text = sample_file.read_text(encoding="utf-8")
    
    # 2. Initialize comparator
    comparator = ChunkingStrategyComparator()
    
    # 3. Run comparison
    print(f"Comparing chunking strategies for: {sample_file.name}")
    print(f"Using embedding backend: {getattr(embedder, '_backend_name', 'mock')}")
    print(f"Total character length: {len(text)}")
    print("-" * 50)
    
    results = comparator.compare(text, chunk_size=500, embedder=embedder)
    
    # 4. Display results
    for strategy, data in results.items():
        print(f"\n[Strategy: {strategy}]")
        print(f" - Chunk Count: {data['count']}")
        print(f" - Avg Chunk Length: {data['avg_length']:.2f} chars")
        
        # Show first 2 chunks as preview
        for i, chunk in enumerate(data['chunks'][:2]):
            preview = chunk.replace("\n", " ")[:80]
            print(f"   Chunk {i+1} preview: {preview}...")

if __name__ == "__main__":
    main()
