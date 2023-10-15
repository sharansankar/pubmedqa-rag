# Self-Qeurying RAG over PubMDQA Dataset
This system implements a self-querying RAG to answer questions over the PubMDQA dataset.
Additionally, this system can be extended to:
1. Any dataset on hugging face
2. Any OpenAI GPT service
3. Multiple levels of self querying

see dataset: https://huggingface.co/datasets/FedML/PubMedQA_instruction

## Quick Start
```
git clone https://github.com/sharansankar/pubmedqa-rag.git
cd pubmedqa-rag

touch .env
# add Open AI API credentials to env file
# OPENAI_ORG_KEY="..."
# OPENAI_API_KEY="..."

# create virtual env
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
cd src

# Run the Program in demo mode
python main.py demo

# OR
# Run in prod mode
python main.py prod
```
## Objective
Design a RAG system that:
1. Can answer questions over any dataset (currently only limited to datasets available on hugging face)
2. Perform self-querying over a dataset
3. track upvotes on LLM responses

## System Design
The system is composed of the following components:
1. `SessionBuilder`: this is responsible for initializing a chat session and orchestrating end-to-end processing of each end-user prompt
2. `OpenAILLMChatClient`: a wrapper around OpenAI Chat Completion endpoint, allowing for easy calling and response parsing of the LLM API
3. `ChromaVectorStore`: wrapper around chroma vector store API. This will be used to store all dataset contextual info
4. `PromptGenerator`: responsible for generating prompts based on available context and user input to then be sent to the LLM API
5. `HuggingFaceDatasetStore`: wrapper around hugging face dataset that will store the dataset in a configurable datastore and can be used for context querying

## How does Self-Querying Work?
When a prompt is submitted to the system, it goes through the following flow:
1. The prompt is fed to the LLM API with no context. The LLM is given and hard-coded example asked to come up with `n` topics to search up in the context db. A sample prompt is shown below:
```
You are a LLM with no context on the following Pubmed-related prompt: {prompt}

You have a Pubmed context database that you can use to query for context. You must output a set of {n} topics separated by commas
to query this context database

As an example:
Prompt: Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?
response: programmed cell death and mitochondria, mitochondria and lace plant leaves, remodelling lace plant leaves

return your response in the form of {n} comma separated strings
```
by default, `n=3`.
2. The LLM response is then parsed by the system, and used to query our `HuggingFaceDatasetStore`, which will retrieve the topics based on embedding similarity.  
3. Using the retrieved context, we then prompt the LLM agent again to answer the initial user prompt. A sample prompt is shown below:
```
You are a medical expert. Answer the following prompt, given the following context to use as your aid

prompt: {prompt}

context:
{context}
```
4. the LLM response is then parsed and surfaced back to the end-user

Benefits of this system:
1. [assumption] The LLM is able to come up with more diverse and contextually important topics to search the datastore. This enables us to get away with simpler embedding models in our vector store.
2. Extendable: this system allows for multiple datasets to be stored and queried by the LLM, allowing for better prompt responses. This design can eventually be extended to all types of queryable data sources (tables, functions, etc.)
3. Dataset agnostic: this design makes the system high performing on any dataset, with any size. We can always guarantee the best context retrieval given the strong contextual understanding of the GPT model.
4. Compute efficient: by relying on short-form topics determined by the LLM, we can reduce the compute required in the embedding-stage. In the future, we can use embed all documents according to their set of topics, allowing for even simpler embeddings.

## Trade Offs Made
The following conscious trade-offs were made:
1. Self-querying over model-experimentation: instead of playing around with different vector stores and embeddings, the decision was made to implement self-querying so that much simpler embeddings could be leveraged.
2. Design patterns over system functionality: instead of implementing a system with expansive functionality (support for multiple dataset type retrieval, historic context lookup, etc.), the decision was made to build out design patterns upfront that could easily extend initial functionality to more use-cases in the future.
3. Type hinting over commenting/docstrings: given the relatively extendable design patterns used, the decision was made to use type hinting as much as possible throughout the codebase. This was done in an effort to make the code as readable as possible without the need for docstrings. This was also done in the hopes of creating a repo that heavily favors more verbose function/variable/class names + type hinting that can be easily understood without the need for any comments.  
