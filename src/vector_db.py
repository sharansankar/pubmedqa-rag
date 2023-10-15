import attr
import chromadb
from datasets import load_dataset
from enum import Enum
from typing import List, Optional

from data_model import QueryResult, DataStoreType

CHROMA_CLIENT = None

def get_chroma_client():
  global CHROMA_CLIENT
  if CHROMA_CLIENT is None:
    CHROMA_CLIENT = chromadb.Client()
  return CHROMA_CLIENT


class ChromaVectorStore:
  def __init__(self, db_name):
    self.chroma_client = get_chroma_client()
    self.db_name = db_name
    self.collection = self.chroma_client.create_collection(db_name)

  def load_data(self, text_list: List[str], ids: Optional[List[str]] = None):
    if ids:
      assert len(ids) == text_list
      text_ids = [str(id) for id in ids]
    else:
      text_ids = [str(i) for i in range(0, len(text_list))]

    self.collection.add(
      ids=text_ids,
      documents=text_list
    )

  def _flatten_documents(self, documents: List) -> List[str]:
    return [text for document in documents for text in document]

  def _parse_response(self, query_response) -> QueryResult:
    return QueryResult(
      raw_text_response="",
      extracted_values=self._flatten_documents(query_response["documents"]),
      metadata=query_response
    )

  def query_store(self, lookup_identifiers: List[str], num_results=1) -> QueryResult:
    chroma_response = self.collection.query(
      query_texts=lookup_identifiers,
      n_results=num_results
    )
    return self._parse_response(chroma_response)


DATA_STORE_TO_CLASS = {
DataStoreType.CHORMA_VECTOR_STORE: ChromaVectorStore
}


class HuggingFaceDatasetStore:
  def __init__(
    self,
    config: "DatasetStoreConfig"
  ):
    self.config= config
    data_store = DATA_STORE_TO_CLASS.get(config.data_store_type)
    self.data_store = data_store(db_name=config.data_store_type.value)
    self.dataset = self._get_hugging_face_ds(config.dataset_name, config.split_name)
    self._load_into_data_store(config.column_identifier, config.num_samples_to_load)

  def _get_hugging_face_ds(self, name, split_name):
    dataset = load_dataset(name, split=split_name)
    return dataset

  def _load_into_data_store(self, col_identifier, sample: Optional[int] = None):
    load_text = self.dataset[col_identifier]

    if sample:
      load_text = load_text[:sample]

    self.data_store.load_data(
      text_list=load_text
    )

  def query_dataset(self, query_texts: List[str], num_results=1) -> QueryResult:
    return self.data_store.query_store(
      query_texts,
      num_results=num_results
    )


if __name__ == "__main__":
  from configs import DEMO_PUBMED_DATASET_CONFIG

  hfds = HuggingFaceDatasetStore(DEMO_PUBMED_DATASET_CONFIG)
  print(hfds.query_dataset(['is this working?']))


