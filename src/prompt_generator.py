from typing import Optional, List

from data_model import QueryContextType, PromptResponseAttributes
from testing import build_test_prompt_response

DEFAULT_PROMPT_WITH_CONTEXT_TEMPLATE = """
You are a medical expert. Answer the following prompt, given the following context to use as your aid

prompt: {prompt}

context:
{context}
"""

QUERY_CONTEXT_TO_TEMPLATE_MAPPING = {
  QueryContextType.PUBMED_CONTEXT: DEFAULT_PROMPT_WITH_CONTEXT_TEMPLATE
}


class PromptGenerator:
  def __init__(
    self,
    query_type: QueryContextType=QueryContextType.PUBMED_CONTEXT,
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
    prompt = self._get_prompt(prompt_response_attrs)
    context_texts = '\n'.join(self._get_context_texts(prompt_response_attrs))
    return self.template.format(prompt=prompt, context=context_texts)


def build_test_response():
  test_pr_attrs = build_test_prompt_response()
  prompt_generator = PromptGenerator()
  print(prompt_generator.build_prompt(test_pr_attrs))

if __name__ == "__main__":
  build_test_response()