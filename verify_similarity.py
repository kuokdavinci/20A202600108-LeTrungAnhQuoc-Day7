import os
from dotenv import load_dotenv
from src.chunking import compute_similarity
from src.embeddings import OpenAIEmbedder, LocalEmbedder, MockEmbedder, EMBEDDING_PROVIDER_ENV

def main():
    load_dotenv()
    
    # 1. Initialize Embedder
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "openai":
        embedder = OpenAIEmbedder()
    elif provider == "local":
        try:
            embedder = LocalEmbedder()
        except ImportError:
            print("Sentence-transformers not installed, falling back to mock.")
            embedder = MockEmbedder()
    else:
        embedder = MockEmbedder()

    print(f"Using embedding backend: {getattr(embedder, 'model_name', 'mock')}")
    print("-" * 80)
    print(f"{'#':<3} | {'Sentence A':<45} | {'Sentence B':<45} | {'Score':<6}")
    print("-" * 80)

    # 2. Define pairs from Section 5 of REPORT.md
    pairs = [
        (
            "Jammed enemies di chuyển nhanh hơn bình thường.", 
            "Kẻ thù bị Jammed có tốc độ di chuyển tăng cao."
        ),
        (
            "EU AI Act quy định về các rủi ro của AI.", 
            "Luật Châu Âu kiểm soát việc triển khai trí tuệ nhân tạo."
        ),
        (
            "Cách nấu mỳ ramen ngon tại nhà.", 
            "Hướng dẫn lập trình Python cơ bản."
        ),
        (
            "Keybullet Kin rơi ra chìa khóa khi bị tiêu diệt.", 
            "Hạ gục Keybullet Kin sẽ nhận được chìa khóa."
        ),
        (
            "Hôm nay trời đẹp.", 
            "Tôi đang học về Vector Database."
        )
    ]

    # 3. Compute and display
    for i, (a, b) in enumerate(pairs, 1):
        emb_a = embedder(a)
        emb_b = embedder(b)
        
        # embedder returns a single list for string input
        score = compute_similarity(emb_a, emb_b)
        
        print(f"{i:<3} | {a[:42]+'...':<45} | {b[:42]+'...':<45} | {score:.3f}")

if __name__ == "__main__":
    main()
