import time

from context_extractors import dataset_context_extractor_from_prompt, ExtractorContext
from data_model import SessionAttriubtes, PromptResponseAttributes, EndUserPrompt, QueryContextType, HumanFeedback
from llm_agent import OpenAILLMChatClient
from prompt_generator import PromptGenerator
from vector_db import HuggingFaceDatasetStore

TEST_DATASET = 'FedML/PubMedQA_instruction'
TEST_SPLIT_NAME = 'train'
TEST_SAMPLES = 100
TEST_COL_IDENTIFIER = 'context'


class SessionBuilder:
  def _gen_session_id(self) -> int:
    return int(time.time())

  def _build_session(self, ):
    session = SessionAttriubtes(
      session_id=self._gen_session_id()
    )
    self.current_session = session

  def __init__(self, ):
    self._build_session()
    self.data_store = HuggingFaceDatasetStore(
      dataset_name=TEST_DATASET,
      split_name=TEST_SPLIT_NAME,
      col_store_identifier=TEST_COL_IDENTIFIER,
      num_samples_to_load=TEST_SAMPLES
    )
    self.llm_agent = OpenAILLMChatClient()
    self.prompt_generator = PromptGenerator()
    self.last_pr_attrs = None

  def create_prompt_response_attrs_with_prompt(self, prompt: str):
    return PromptResponseAttributes(
      end_user_prompt=EndUserPrompt(
        prompt_text=prompt
      )
    )

  def _update_session_state(self, pr_attrs: PromptResponseAttributes):
    self.latest_pr_attrs = pr_attrs
    self.current_session.PromptResponseAttributesList.append(pr_attrs)

  def process_user_prompt(self, prompt: str):
    # 1 create prompt obj with end user prompt
    pr_attrs = self.create_prompt_response_attrs_with_prompt(prompt)

    # 2 hydrate context for prompt
    pr_attrs.context_querying_results[QueryContextType.PUBMED_CONTEXT] = dataset_context_extractor_from_prompt(
      pr_attrs=pr_attrs,
      extractor_context=ExtractorContext(
        dataset_store=self.data_store
      )
    )

    # 3 Build LLM Agent Prompt
    llm_agent_prompt = self.prompt_generator.build_prompt(pr_attrs)

    # 4 Get Prompt
    pr_attrs.llm_prompt_response = self.llm_agent.get_response_from_prompt(llm_agent_prompt)

    # 5 Update Session state
    self._update_session_state(pr_attrs)

  def response_is_upvoted(self):
    self.latest_pr_attrs.human_feedback = HumanFeedback(is_upvoted=True)
