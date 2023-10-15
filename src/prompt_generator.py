from typing import Optional, List

from data_model import QueryContextType, PromptResponseAttributes
from prompt_response_attrs_utils import get_end_user_prompt, get_query_context_extracted_values
from testing import build_test_prompt_response

DEFAULT_PROMPT_WITH_CONTEXT_TEMPLATE = """
You are a medical expert. Answer the following prompt, given the following context to use as your aid

prompt: {prompt}

context:
{context}
"""

DEFAULT_TOPIC_SELF_QUERY_TEMPLATE = """
You are a LLM with no context on the following Pubmed-related prompt: {prompt}

You have a Pubmed context database that you can use to query for context. You must output a set of {n} topics separated by commas 
to query this context database

As an example:
Prompt: Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?
response: programmed cell death and mitochondria, mitochondria and lace plant leaves, remodelling lace plant leaves 

return your response in the form of {n} comma separated strings 
"""


QUERY_CONTEXT_TO_TEMPLATE_MAPPING = {
  QueryContextType.PUBMED_CONTEXT: DEFAULT_PROMPT_WITH_CONTEXT_TEMPLATE,
  QueryContextType.PUBMED_TOPIC_SELF_QUERY: DEFAULT_TOPIC_SELF_QUERY_TEMPLATE
}


def generate_context_llm_prompt(prompt_response_attrs: PromptResponseAttributes, template: str) -> str:
  prompt = get_end_user_prompt(prompt_response_attrs)
  context_texts = '\n'.join(get_query_context_extracted_values(
    prompt_response_attrs,
    query_type=QueryContextType.PUBMED_CONTEXT
  ))
  return template.format(prompt=prompt, context=context_texts)


def generate_llm_topic_self_querying_prompt(prompt_response_attrs: PromptResponseAttributes, template: str, n=3):
  prompt = get_end_user_prompt(prompt_response_attrs)
  return template.format(prompt=prompt, n=n)


QUERY_TYPE_TO_GENERATOR = {
  QueryContextType.PUBMED_CONTEXT: generate_context_llm_prompt,
  QueryContextType.PUBMED_TOPIC_SELF_QUERY: generate_llm_topic_self_querying_prompt
}


class PromptGenerator:
  def __init__(
    self,
    query_type: QueryContextType = QueryContextType.PUBMED_CONTEXT,
    template_override: Optional[str] = None
  ):
    self.query_type = query_type
    if template_override:
      self.template = template_override
    else:
      self.template = QUERY_CONTEXT_TO_TEMPLATE_MAPPING.get(query_type)

  def _get_prompt(self, prompt_response_attrs):
    if prompt_response_attrs.end_user_prompt:
      return prompt_response_attrs.end_user_prompt.prompt_text

  def _get_context_texts(self, prompt_response_attrs) -> List[str]:
    if prompt_response_attrs.context_querying_results:
      query_result = prompt_response_attrs.context_querying_results.get(self.query_type, None)
      if query_result:
        return query_result.extracted_values
      return []

  def build_prompt(self, prompt_response_attrs: PromptResponseAttributes) -> str:
    prompt_generator = QUERY_TYPE_TO_GENERATOR.get(self.query_type)
    return prompt_generator(prompt_response_attrs, self.template)


def build_test_response():
  test_pr_attrs = build_test_prompt_response()
  prompt_generator = PromptGenerator(query_type=QueryContextType.PUBMED_TOPIC_SELF_QUERY)
  print(prompt_generator.build_prompt(test_pr_attrs))


if __name__ == "__main__":
  build_test_response()