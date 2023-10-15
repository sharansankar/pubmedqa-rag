import attr
from typing import Optional, List

from data_model import DataStoreType, LLMInstanceTypes, QueryContextType


@attr.s
class DatasetStoreConfig:
  dataset_name: str = attr.ib()
  split_name: str = attr.ib()
  column_identifier: str = attr.ib()
  data_store_type: DataStoreType = attr.ib(default=DataStoreType.CHORMA_VECTOR_STORE)
  num_samples_to_load: Optional[int] = attr.ib(default=None)


DEMO_PUBMED_DATASET_CONFIG = DatasetStoreConfig(
  dataset_name='FedML/PubMedQA_instruction',
  split_name='train',
  column_identifier="context",
  num_samples_to_load=100
)

PROD_PUBMED_DATASET_CONFIG = DatasetStoreConfig(
  dataset_name='FedML/PubMedQA_instruction',
  split_name='train',
  column_identifier="context",
)


@attr.s
class LLMConfig:
  model_name: LLMInstanceTypes = attr.ib(default=LLMInstanceTypes.GPT_3_5_TURBO)
  max_tokens: int = attr.ib(default=2000)
  temperature: float = attr.ib(default=0.7)


DEMO_LLM_CONFIG = LLMConfig()


@attr.s
class SessionConfig:
  data_store_config: DatasetStoreConfig = attr.ib()
  llm_client_config: LLMConfig = attr.ib()
  extraction_stages: List[QueryContextType] = attr.ib()
  debugging: bool = attr.ib(default=False)


DEMO_SESSION_CONFIG = SessionConfig(
  data_store_config=DEMO_PUBMED_DATASET_CONFIG,
  llm_client_config=DEMO_LLM_CONFIG,
  extraction_stages=[
    QueryContextType.PUBMED_TOPIC_SELF_QUERY,
    QueryContextType.PUBMED_CONTEXT
  ],
  debugging=False
)

PROD_SESSION_CONFIG = SessionConfig(
  data_store_config=PROD_PUBMED_DATASET_CONFIG,
  llm_client_config=DEMO_LLM_CONFIG,
  extraction_stages=[
    QueryContextType.PUBMED_TOPIC_SELF_QUERY,
    QueryContextType.PUBMED_CONTEXT
  ],
  debugging=False
)