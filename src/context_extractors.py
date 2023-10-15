import attr
from typing import Optional, List

from data_model import PromptResponseAttributes
from vector_db import HuggingFaceDatasetStore


@attr.s
class ExtractorContext:
  dataset_store: HuggingFaceDatasetStore = attr.ib()


def dataset_context_extractor_from_prompt(pr_attrs: PromptResponseAttributes, extractor_context: ExtractorContext):
  query_list = [
    pr_attrs.end_user_prompt.prompt_text
  ]
  dataset_store = extractor_context.dataset_store
  return dataset_store.query_dataset(query_list)
