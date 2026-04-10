import os
from typing import Optional, Callable
from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.
    Powered by OpenAI's GPT models (e.g. gpt-4o-mini) or local Llama via Ollama.
    """

    def __init__(
        self, 
        store: EmbeddingStore, 
        llm_fn: Optional[Callable[[str], str]] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> None:
        import os
        self.store = store
        self.llm_fn = llm_fn
        
        # Determine provider and model from environment
        provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
        self.model_name = model_name or os.getenv("LLM_MODEL", "gpt-4o-mini")
        
        # Auto-configure base_url and api_key
        if not base_url:
            if provider == "openai":
                base_url = "https://api.openai.com/v1"
                api_key = os.getenv("OPENAI_API_KEY")
            else:
                # Default to local Ollama
                base_url = "http://localhost:11434/v1"
                api_key = "ollama"
        else:
            api_key = os.getenv("OPENAI_API_KEY", "ollama")

        # Initialize OpenAI client only if no standard llm_fn is provided
        if not self.llm_fn:
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    base_url=base_url,
                    api_key=api_key, 
                )
            except ImportError:
                print("\n[WARNING] 'openai' module not found. Falling back to mock LLM responses.")
                self.llm_fn = lambda prompt: f"[MOCK ANSWER] Response for: {prompt[:50]}..."
                self.client = None
        else:
            self.client = None

    def answer(self, question: str, top_k: int = 3, category_filter: Optional[str] = None) -> str:
        """
        Retrieval-augmented generation (RAG) implementation.
        """
        # 1. Retrieve relevant chunks
        if category_filter:
            results = self.store.search_with_filter(
                query=question, 
                filter_dict={"category": category_filter}, 
                top_k=top_k
            )
        else:
            results = self.store.search(query=question, top_k=top_k)

        if not results:
            return "Tôi không tìm thấy thông tin liên quan trong cơ sở kiến thức để trả lời câu hỏi của bạn."

        # 2. Build the context string
        context_parts = []
        for i, res in enumerate(results, start=1):
            meta = res.get("metadata", {})
            source = meta.get("source", "N/A")
            category = meta.get("category", "General")
            url = meta.get("source_url", "N/A")
            
            context_parts.append(
                f"--- Đoạn trích {i} [Nguồn: {source}] [Chủ đề: {category}] [URL: {url}] ---\n"
                f"{res.get('content', '')}"
            )
        
        context = "\n\n".join(context_parts)

        # 3. Construct the prompt
        system_prompt = (
            "Bạn là một trợ lý AI hữu ích. Hãy sử dụng các đoạn trích từ cơ sở kiến thức được cung cấp "
            "để trả lời câu hỏi của người dùng một cách chính xác và trung thực. "
            "Nếu thông tin không có trong cơ sở kiến thức, hãy nói rằng bạn không biết, "
            "đừng tự bịa ra câu trả lời. Luôn trả lời bằng tiếng Việt."
        )
        
        user_message = f"Dựa vào các thông tin sau đây:\n\n{context}\n\nCâu hỏi: {question}"

        # 4. Use injected llm_fn if provided (for tests/custom logic)
        if self.llm_fn:
            return self.llm_fn(f"{system_prompt}\n\n{user_message}")

        # 5. Call OpenAI / Ollama API
        try:
            if not self.client:
                return "Error: LLM client not initialized."
                
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,
            )
            return response.choices[0].message.content or "Error: Empty response from model."
        except Exception as e:
            return f"Lỗi gọi LLM: {str(e)}"
