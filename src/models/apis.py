from dataclasses import dataclass

@dataclass
class MistralAPI:
    api_key: str
    endpoint_url: str
    model: str
    temperature: float
    max_tokens: int
    top_k: int
    top_p: float
    min_p: float

@dataclass
class MistralAgent:
    api_key: str
    endpoint_url: str
    agent_id: str
    max_tokens: int

@dataclass
class ElevenLabsAPI:
    api_key: str
    endpoint_url: str
    model_id: str
    voice_id: str
    output_format: str