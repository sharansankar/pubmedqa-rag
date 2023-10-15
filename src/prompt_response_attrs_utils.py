from typing import Optional, List

from data_model import QueryContextType, PromptResponseAttributes


def get_end_user_prompt(prompt_response_attrs: PromptResponseAttributes) -> Optional[str]:
  if prompt_response_attrs.end_user_prompt:
    return prompt_response_attrs.end_user_prompt.prompt_text


def get_query_context_extracted_values(
  prompt_response_attrs: PromptResponseAttributes,
  query_type: QueryContextType
) -> List[str]:
  if prompt_response_attrs.context_querying_results:
    query_result = prompt_response_attrs.context_querying_results.get(query_type, None)
    if query_result:
      return query_result.extracted_values
    return []
