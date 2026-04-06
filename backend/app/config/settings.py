"""
Synthetic Data Engine Settings
Unified configuration from environment variables
"""

import os
from dotenv import load_dotenv

# Load project root .env
project_root_env = os.path.join(os.path.dirname(__file__), '../../../.env')
if os.path.exists(project_root_env):
    load_dotenv(project_root_env, override=True)
else:
    load_dotenv(override=True)


class Settings:
    """Engine configuration"""

    # Flask
    SECRET_KEY = os.environ.get('SECRET_KEY', 'synthetic-engine-secret')
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    JSON_AS_ASCII = False

    # Primary LLM
    LLM_API_KEY = os.environ.get('LLM_API_KEY')
    LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'https://api.openai.com/v1')
    LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME', 'gpt-4o-mini')

    # Boost LLM (optional — faster/cheaper for bulk work)
    LLM_BOOST_API_KEY = os.environ.get('LLM_BOOST_API_KEY')
    LLM_BOOST_BASE_URL = os.environ.get('LLM_BOOST_BASE_URL')
    LLM_BOOST_MODEL_NAME = os.environ.get('LLM_BOOST_MODEL_NAME')

    # Zep Memory
    ZEP_API_KEY = os.environ.get('ZEP_API_KEY')

    # File upload
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../../uploads')
    ALLOWED_EXTENSIONS = {'pdf', 'md', 'txt', 'csv', 'json', 'py', 'js', 'ts', 'java', 'go', 'rs'}

    # Engine defaults
    CHUNK_TOKEN_SIZE = int(os.environ.get('CHUNK_TOKEN_SIZE', '1500'))
    CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', '200'))
    MAX_CONCURRENT_AGENTS = int(os.environ.get('MAX_CONCURRENT_AGENTS', '5'))
    QUALITY_THRESHOLD = float(os.environ.get('QUALITY_THRESHOLD', '0.7'))
    NOVELTY_THRESHOLD = float(os.environ.get('NOVELTY_THRESHOLD', '0.6'))

    # Agent settings
    AGENT_MAX_RETRIES = int(os.environ.get('AGENT_MAX_RETRIES', '3'))
    AGENT_TEMPERATURE = float(os.environ.get('AGENT_TEMPERATURE', '0.7'))
    AGENT_TIMEOUT = int(os.environ.get('AGENT_TIMEOUT', '120'))

    # Cost tracking
    COST_TRACKING_ENABLED = os.environ.get('COST_TRACKING', 'True').lower() == 'true'
    MAX_BUDGET_USD = float(os.environ.get('MAX_BUDGET_USD', '10.0'))

    # Export
    SUPPORTED_EXPORT_FORMATS = ['json', 'csv', 'jsonl', 'parquet']

    @classmethod
    def validate(cls):
        """Validate required config"""
        errors = []
        if not cls.LLM_API_KEY:
            errors.append("LLM_API_KEY is required")
        return errors

    @classmethod
    def get_llm_config(cls, use_boost: bool = False) -> dict:
        """Get LLM config dict for client initialization"""
        if use_boost and cls.LLM_BOOST_API_KEY:
            return {
                "api_key": cls.LLM_BOOST_API_KEY,
                "base_url": cls.LLM_BOOST_BASE_URL,
                "model": cls.LLM_BOOST_MODEL_NAME,
            }
        return {
            "api_key": cls.LLM_API_KEY,
            "base_url": cls.LLM_BASE_URL,
            "model": cls.LLM_MODEL_NAME,
        }
