from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot, compute_similarity
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
        chunker: Any = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._chunker = chunker
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb

            # Initialize chromadb client + collection
            client = chromadb.Client()
            self._collection = client.get_or_create_collection(
                name=self._collection_name,
                embedding_function=lambda texts: [self._embedding_fn(t) for t in texts]
            )
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        # Build a normalized stored record for one document
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": doc.metadata or {},
            "embedding": self._embedding_fn(doc.content)
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        # Run in-memory similarity search over provided records
        if not records:
            return []
            
        query_vec = self._embedding_fn(query)
        scored_records = []
        
        for rec in records:
            score = compute_similarity(query_vec, rec["embedding"])
            scored_records.append({
                "id": rec["id"],
                "content": rec["content"],
                "metadata": rec["metadata"],
                "score": score
            })
            
        # Sort by score descending
        scored_records.sort(key=lambda x: x["score"], reverse=True)
        return scored_records[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        all_chunks: list[Document] = []
        
        if self._chunker:
            for doc in docs:
                chunks = self._chunker.chunk(doc.content)
                for i, chunk_text in enumerate(chunks):
                    # Create a new Document for each chunk, preserving metadata
                    chunk_doc = Document(
                        id=f"{doc.id}_chunk_{i}",
                        content=chunk_text,
                        metadata={**doc.metadata, "doc_id": doc.id}
                    )
                    all_chunks.append(chunk_doc)
        else:
            all_chunks = docs

        if self._use_chroma and self._collection:
            ids = [doc.id if doc.id else f"id_{i}_{self._next_index}" for i, doc in enumerate(all_chunks)]
            self._next_index += len(all_chunks)
            documents = [doc.content for doc in all_chunks]
            metadatas = [doc.metadata or {} for doc in all_chunks]
            
            # Batch embed all documents at once
            embeddings = self._embedding_fn(documents)
            
            self._collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
        else:
            # Batch embed all documents at once for in-memory store too
            documents = [doc.content for doc in all_chunks]
            print(f"  [EmbeddingStore] Embedding {len(documents)} final chunks...")
            embeddings = self._embedding_fn(documents)
            
            for i, doc in enumerate(all_chunks):
                record = {
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata or {},
                    "embedding": embeddings[i]
                }
                self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma and self._collection:
            query_vec = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_vec],
                n_results=top_k
            )
            # Reformat chroma results to match internal format
            formatted = []
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    formatted.append({
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": results["distances"][0][i]  # Chroma uses distance, usually smaller is better
                    })
            return formatted
        else:
            return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if self._use_chroma and self._collection:
            query_vec = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_vec],
                n_results=top_k,
                where=metadata_filter
            )
            # Reformat similar to search()
            formatted = []
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    formatted.append({
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": results["distances"][0][i]
                    })
            return formatted
        else:
            filtered_records = []
            for rec in self._store:
                match = True
                if metadata_filter:
                    for k, v in metadata_filter.items():
                        if rec["metadata"].get(k) != v:
                            match = False
                            break
                if match:
                    filtered_records.append(rec)
            return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma and self._collection:
            # ChromaDB doesn't usually have a simple way to delete by partial metadata
            # but we assume doc_id is in metadata or is the id.
            count_before = self._collection.count()
            self._collection.delete(where={"doc_id": doc_id})
            return self._collection.count() < count_before
        else:
            original_len = len(self._store)
            self._store = [r for r in self._store if r["id"] != doc_id and r["metadata"].get("doc_id") != doc_id]
            return len(self._store) < original_len
