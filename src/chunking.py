from __future__ import annotations

import math
import re
from typing import Callable, Any, Optional


class SemanticChunker:
    """
    Groups sentences into chunks based on semantic similarity using embeddings.
    A new chunk starts whenever the similarity between the current and previous 
    sentence groups falls below a specified threshold.
    """

    def __init__(
        self, 
        embedding_fn: Callable[[str], list[float]], 
        threshold: float = 0.5,
        max_chunk_size: int = 1000
    ) -> None:
        self.embedding_fn = embedding_fn
        self.threshold = threshold
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
            
        # 1. Split into sentences (simple regex-based split)
        pattern = r'(?<=[.!?])\s+|(?<=\.\n)'
        sentences = [s.strip() for s in re.split(pattern, text) if s.strip()]
        if not sentences:
            return []
            
        # 2. Embed all sentences (Batch)
        print(f"  [SemanticChunker] Embedding {len(sentences)} sentences...")
        sentence_embeddings = self.embedding_fn(sentences)
        
        # 3. Group sentences by similarity
        chunks = []
        current_chunk_sentences = [sentences[0]]
        
        for i in range(1, len(sentences)):
            # Calculate similarity with the previous sentence
            similarity = self._compute_similarity(sentence_embeddings[i], sentence_embeddings[i-1])
            
            # Heuristic: Start new chunk if similarity is low OR current chunk is getting too big
            current_len = sum(len(s) + 1 for s in current_chunk_sentences) - 1
            
            if similarity < self.threshold or current_len > self.max_chunk_size:
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = [sentences[i]]
            else:
                current_chunk_sentences.append(sentences[i])
        
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
            
        return chunks

    def _compute_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        dot_prod = sum(x * y for x, y in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(x * x for x in vec_a))
        norm_b = math.sqrt(sum(x * x for x in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_prod / (norm_a * norm_b)


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=\.\n)'
        sentences = [s.strip() for s in re.split(pattern, text) if s.strip()]
        
        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk_sentences = sentences[i : i + self.max_sentences_per_chunk]
            chunks.append(" ".join(chunk_sentences))
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size:
            return [current_text]
        
        if not remaining_separators:
            # If no more separators, just split by chunk_size
            return [current_text[i:i+self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        separator = remaining_separators[0]
        next_separators = remaining_separators[1:]
        
        # Split by the current separator
        if separator == "":
            # Character split
            parts = list(current_text)
        else:
            parts = current_text.split(separator)
            # Re-add the separator to parts except the last one (optional, but often preferred)
            # However, standard RecursiveCharacterTextSplitter often just splits.
            # Let's stick to simple split and join for now.
        
        final_chunks: list[str] = []
        current_part = ""
        
        for part in parts:
            # If a single part is still too long, we must split it recursively
            if len(part) > self.chunk_size:
                # If we had some accumulated current_part, add it first
                if current_part:
                    final_chunks.append(current_part)
                    current_part = ""
                
                # Recursively split the oversized part
                final_chunks.extend(self._split(part, next_separators))
            else:
                # Try to accumulate parts
                separator_len = len(separator) if current_part else 0
                if len(current_part) + separator_len + len(part) <= self.chunk_size:
                    if current_part:
                        current_part += separator
                    current_part += part
                else:
                    if current_part:
                        final_chunks.append(current_part)
                    current_part = part
        
        if current_part:
            final_chunks.append(current_part)
            
        return final_chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    dot_prod = _dot(vec_a, vec_b)
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(x * x for x in vec_b))
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_prod / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200, embedder: Optional[Callable] = None) -> dict:
        strategies = {
            'fixed_size': FixedSizeChunker(chunk_size=chunk_size, overlap=chunk_size // 10),
            'by_sentences': SentenceChunker(max_sentences_per_chunk=3),
            'recursive': RecursiveChunker(chunk_size=chunk_size),
        }
        
        if embedder:
            strategies['semantic'] = SemanticChunker(embedding_fn=embedder, threshold=0.5)
        
        results = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            if not chunks:
                results[name] = {'count': 0, 'avg_length': 0, 'chunks': []}
                continue
                
            lengths = [len(c) for c in chunks]
            results[name] = {
                'count': len(chunks),
                'avg_length': sum(lengths) / len(chunks),
                'chunks': chunks
            }
        return results
