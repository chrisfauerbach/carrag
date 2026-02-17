from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    es_url: str = "http://elasticsearch:9200"
    ollama_url: str = "http://ollama:11434"
    embedding_model: str = "nomic-embed-text"
    llm_model: str = "llama3.2"
    chunk_size: int = 2000
    chunk_overlap: int = 200
    es_index: str = "carrag_chunks"
    es_chat_index: str = "carrag_chats"
    es_metrics_index: str = "carrag_metrics"
    es_prompts_index: str = "carrag_prompts"


settings = Settings()
