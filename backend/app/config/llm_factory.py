"""
LLM Backend Factory
Creates OpenAI-compatible clients with primary/boost fallback
"""

from openai import OpenAI
from .settings import Settings


class LLMFactory:
    """
    Factory for creating LLM clients.
    Supports primary and boost (cheaper/faster) backends.
    """

    _primary_client: OpenAI | None = None
    _boost_client: OpenAI | None = None

    @classmethod
    def get_client(cls, use_boost: bool = False) -> OpenAI:
        """Get an OpenAI-compatible client"""
        if use_boost and Settings.LLM_BOOST_API_KEY:
            if cls._boost_client is None:
                config = Settings.get_llm_config(use_boost=True)
                cls._boost_client = OpenAI(
                    api_key=config["api_key"],
                    base_url=config["base_url"],
                )
            return cls._boost_client

        if cls._primary_client is None:
            config = Settings.get_llm_config()
            cls._primary_client = OpenAI(
                api_key=config["api_key"],
                base_url=config["base_url"],
            )
        return cls._primary_client

    @classmethod
    def get_model(cls, use_boost: bool = False) -> str:
        """Get model name"""
        config = Settings.get_llm_config(use_boost=use_boost)
        return config["model"]

    @classmethod
    def complete(
        cls,
        messages: list[dict],
        use_boost: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> str:
        """Simple completion — returns text content"""
        client = cls.get_client(use_boost=use_boost)
        model = cls.get_model(use_boost=use_boost)

        kwargs = {
            "model": model,
            "messages": messages,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format

        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    @classmethod
    def complete_json(
        cls,
        messages: list[dict],
        use_boost: bool = False,
        temperature: float | None = None,
    ) -> dict | list:
        """Completion with JSON response parsing"""
        import json

        content = cls.complete(
            messages=messages,
            use_boost=use_boost,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return json.loads(content)

    @classmethod
    def reset(cls):
        """Reset cached clients"""
        cls._primary_client = None
        cls._boost_client = None
