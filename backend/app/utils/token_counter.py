"""
Token Counter — accurate token counting for chunking and cost estimation
"""

import tiktoken
from typing import Optional


class TokenCounter:
    """Thread-safe token counter using tiktoken"""

    _encodings: dict[str, tiktoken.Encoding] = {}

    @classmethod
    def _get_encoding(cls, model: str = "gpt-4o") -> tiktoken.Encoding:
        """Get or cache encoding for a model"""
        if model not in cls._encodings:
            try:
                cls._encodings[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fallback to cl100k_base for unknown models
                cls._encodings[model] = tiktoken.get_encoding("cl100k_base")
        return cls._encodings[model]

    @classmethod
    def count(cls, text: str, model: str = "gpt-4o") -> int:
        """Count tokens in text"""
        encoding = cls._get_encoding(model)
        return len(encoding.encode(text))

    @classmethod
    def truncate(cls, text: str, max_tokens: int, model: str = "gpt-4o") -> str:
        """Truncate text to max tokens"""
        encoding = cls._get_encoding(model)
        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return encoding.decode(tokens[:max_tokens])

    @classmethod
    def split_by_tokens(
        cls,
        text: str,
        chunk_size: int = 1500,
        overlap: int = 200,
        model: str = "gpt-4o",
    ) -> list[str]:
        """Split text into token-based chunks with overlap"""
        encoding = cls._get_encoding(model)
        tokens = encoding.encode(text)

        if len(tokens) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(encoding.decode(chunk_tokens))
            if end >= len(tokens):
                break
            start = end - overlap

        return chunks

    @classmethod
    def estimate_cost(
        cls,
        input_tokens: int,
        output_tokens: int,
        model: str = "gpt-4o",
    ) -> float:
        """Estimate cost in USD (rough estimates)"""
        # Pricing per 1M tokens (approximate)
        pricing = {
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
            "qwen-plus": {"input": 0.40, "output": 1.20},
            "qwen-turbo": {"input": 0.06, "output": 0.06},
            "deepseek-chat": {"input": 0.14, "output": 0.28},
        }

        # Find matching pricing
        rates = None
        for key, val in pricing.items():
            if key in model.lower():
                rates = val
                break

        if not rates:
            # Default fallback
            rates = {"input": 1.00, "output": 3.00}

        cost = (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000
        return round(cost, 6)
