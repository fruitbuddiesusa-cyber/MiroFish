"""
Chunker — token-aware text chunking for large inputs
Supports multiple strategies: fixed, semantic, recursive
"""

from typing import Any
from ..utils.token_counter import TokenCounter


class Chunker:
    """
    Smart text chunking with multiple strategies.
    """

    @staticmethod
    def fixed_token_split(
        text: str,
        chunk_size: int = 1500,
        overlap: int = 200,
        model: str = "gpt-4o",
    ) -> list[dict]:
        """
        Fixed-size token-based chunks.
        Returns list of {"text": str, "index": int, "token_count": int}
        """
        raw_chunks = TokenCounter.split_by_tokens(text, chunk_size, overlap, model)
        return [
            {
                "text": chunk,
                "index": i,
                "token_count": TokenCounter.count(chunk, model),
                "char_count": len(chunk),
            }
            for i, chunk in enumerate(raw_chunks)
        ]

    @staticmethod
    def paragraph_split(
        text: str,
        max_chunk_size: int = 1500,
        model: str = "gpt-4o",
    ) -> list[dict]:
        """
        Split by paragraphs, merging small ones.
        Preserves paragraph boundaries.
        """
        paragraphs = text.split('\n\n')
        chunks = []
        current = []
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = TokenCounter.count(para, model)

            if current_tokens + para_tokens > max_chunk_size and current:
                chunk_text = '\n\n'.join(current)
                chunks.append({
                    "text": chunk_text,
                    "index": len(chunks),
                    "token_count": current_tokens,
                    "char_count": len(chunk_text),
                    "strategy": "paragraph",
                })
                current = [para]
                current_tokens = para_tokens
            else:
                current.append(para)
                current_tokens += para_tokens

        if current:
            chunk_text = '\n\n'.join(current)
            chunks.append({
                "text": chunk_text,
                "index": len(chunks),
                "token_count": current_tokens,
                "char_count": len(chunk_text),
                "strategy": "paragraph",
            })

        return chunks

    @staticmethod
    def code_split(
        text: str,
        max_chunk_size: int = 1000,
        model: str = "gpt-4o",
    ) -> list[dict]:
        """
        Split code by function/class boundaries.
        Keeps logical units together.
        """
        import re

        # Split on function/class definitions
        pattern = r'((?:^|\n)(?:(?:def |class |function |const \w+ = |export (?:default )?(?:function|class|const))|/\*\*|\* @))'

        parts = re.split(pattern, text)
        chunks = []
        current = ""
        current_tokens = 0

        for part in parts:
            part = part.strip()
            if not part:
                continue

            part_tokens = TokenCounter.count(part, model)

            if current_tokens + part_tokens > max_chunk_size and current:
                chunks.append({
                    "text": current.strip(),
                    "index": len(chunks),
                    "token_count": current_tokens,
                    "char_count": len(current),
                    "strategy": "code",
                })
                current = part
                current_tokens = part_tokens
            else:
                current += '\n' + part if current else part
                current_tokens += part_tokens

        if current.strip():
            chunks.append({
                "text": current.strip(),
                "index": len(chunks),
                "token_count": current_tokens,
                "char_count": len(current),
                "strategy": "code",
            })

        return chunks if chunks else Chunker.fixed_token_split(text, max_chunk_size, model=model)

    @staticmethod
    def semantic_split(
        text: str,
        max_chunk_size: int = 1500,
        model: str = "gpt-4o",
    ) -> list[dict]:
        """
        Split on semantic boundaries (headers, topic shifts).
        Good for markdown and structured documents.
        """
        import re

        # Split on headers
        parts = re.split(r'(?=^#{1,3}\s)', text, flags=re.MULTILINE)

        chunks = []
        current = ""
        current_tokens = 0

        for part in parts:
            part = part.strip()
            if not part:
                continue

            part_tokens = TokenCounter.count(part, model)

            if current_tokens + part_tokens > max_chunk_size and current:
                chunks.append({
                    "text": current.strip(),
                    "index": len(chunks),
                    "token_count": current_tokens,
                    "char_count": len(current),
                    "strategy": "semantic",
                })
                current = part
                current_tokens = part_tokens
            else:
                current += '\n\n' + part if current else part
                current_tokens += part_tokens

        if current.strip():
            chunks.append({
                "text": current.strip(),
                "index": len(chunks),
                "token_count": current_tokens,
                "char_count": len(current),
                "strategy": "semantic",
            })

        return chunks if chunks else Chunker.paragraph_split(text, max_chunk_size, model)

    @staticmethod
    def auto_split(
        text: str,
        data_type: str = "text",
        max_chunk_size: int = 1500,
        model: str = "gpt-4o",
    ) -> list[dict]:
        """Auto-select best splitting strategy based on data type"""
        if data_type == "code":
            return Chunker.code_split(text, max_chunk_size, model)
        elif data_type in ("markdown", "structured"):
            return Chunker.semantic_split(text, max_chunk_size, model)
        elif data_type == "text":
            return Chunker.paragraph_split(text, max_chunk_size, model)
        else:
            return Chunker.fixed_token_split(text, max_chunk_size, model=model)

    @staticmethod
    def merge_chunks(chunks: list[dict], separator: str = "\n\n") -> str:
        """Merge chunks back into full text"""
        return separator.join(c["text"] for c in chunks)
