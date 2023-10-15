from data_model import *


def build_test_prompt_response() -> PromptResponseAttributes:
  return PromptResponseAttributes(
    end_user_prompt=EndUserPrompt(
      prompt_text="what is love?"
    ),
    context_querying_results={
      QueryContextType.PUBMED_CONTEXT: QueryResult(
        raw_text_response="",
        extracted_values=["baby dont hurt me"]
      )
    }
  )
