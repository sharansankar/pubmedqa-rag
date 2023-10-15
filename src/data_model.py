import attr
from enum import Enum
from typing import Optional, Dict, List


@attr.s
class EndUserPrompt:
  prompt_text: str = attr.ib()
  metadata: Dict = attr.ib(default={})


class ContextDatasets(Enum):
  PUBMED_CONTEXT_DATASET = 1
  # HISTORIC_PROMPT_RESPONSE = 2


class QueryContextType(Enum):
  PUBMED_CONTEXT = 1
  PUBMED_TOPIC_SELF_QUERY = 2
  # HISTORIC_PROMPT_CONTEXT


@attr.s
class QueryResult:
  raw_text_response: str = attr.ib()
  extracted_values: Optional[List[str]] = attr.ib(default=None)
  metadata: Dict = attr.ib(default={})


@attr.s
class LLMPromptResponse:
  text_response: str = attr.ib()
  metadata: Dict = attr.ib(default={})


@attr.s
class HumanFeedback:
  is_upvoted: bool = attr.ib()


@attr.s
class PromptResponseAttributes:
  end_user_prompt: Optional[EndUserPrompt] = attr.ib(default=None)
  context_querying_results: Dict[QueryContextType, QueryResult] = attr.ib(default={})
  llm_prompt_response: Optional[LLMPromptResponse] = attr.ib(default=None)
  human_feedback: Optional[HumanFeedback] = attr.ib(default=None)


@attr.s
class SessionAttriubtes:
  session_id: int = attr.ib()
  PromptResponseAttributesList: List[PromptResponseAttributes] = attr.ib(default=[])


class DataStoreType(Enum):
  CHORMA_VECTOR_STORE = "chroma_vector_store"


class LLMInstanceTypes(Enum):
  GPT_3_5_TURBO = "gpt-3.5-turbo"
  GPT_3_5_TURBO_16K = "gpt-3.5-turbo-16k"
  GPT_4 = "gpt-4"
  GPT_4_32K = "gpt-4-32k"


if __name__ == "__main__":
  print("hello world")