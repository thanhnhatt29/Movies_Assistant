import os
import json
from time import time
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from elasticsearch import Elasticsearch

load_dotenv()

ELASTIC_URL = os.getenv("ELASTIC_URL","http://elasticsearch:9200")
INDEX_NAME = os.getenv("INDEX_NAME", "movies")
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

prompt_template = """
You're a movie assistant. Answer the QUESTION based on the CONTEXT from our movies data.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
""".strip()


entry_template = """
title: {title}
genres: {genres}
original_language: {original_language}
overview: {overview}
popularity: {popularity}
production_companies: {production_companies} 
release_date: {release_date}
budget: {budget}
revenue: {revenue}
runtime: {runtime}
status: {status}
tagline: {tagline}
vote_average: {vote_average}
vote_count: {vote_count}
credits: {credits}
keywords: {keywords}
""".strip()

evaluation_prompt_template = """
You are an expert evaluator for a RAG system.
Your task is to analyze the relevance of the generated answer to the given question.
Based on the relevance of the generated answer, you will classify it
as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()


def elastic_search(query):

    es_client = Elasticsearch(ELASTIC_URL)
    
    search_query = {
        "size": 10,
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["title^3", "description^2", "overview^1.5", "genres", "keywords"],
                "type": "best_fields",
                "fuzziness": "AUTO"
            }
        }
    }

    response = es_client.search(index=INDEX_NAME, body=search_query)

    result_docs = [hit['_source'] for hit in response['hits']['hits']]

    return result_docs


def build_prompt(query, search_results):
    context = ""

    for doc in search_results:
        context = context + entry_template.format(**doc) + "\n\n"

    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt


def llm(prompt, model='mistralai/Mixtral-8x7B-Instruct-v0.1'):
    client = InferenceClient(
        model,
        token=HUGGINGFACE_TOKEN,
    )

    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    
    answer = response.choices[0].message.content
    
    token_stats = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }
    
    return answer, token_stats


def evaluate_relevance(question, answer):
    prompt = evaluation_prompt_template.format(question=question, answer=answer)
    evaluation, tokens = llm(prompt)

    try:
        json_eval = json.loads(evaluation)
        return json_eval, tokens
    except json.JSONDecodeError:
        result = {"Relevance": "UNKNOWN", "Explanation": "Failed to parse evaluation"}
        return result, tokens

def rag(query, model='mistralai/Mixtral-8x7B-Instruct-v0.1'):
    print(ELASTIC_URL)
    t0 = time()

    search_results = elastic_search(query)
    prompt = build_prompt(query, search_results)
    answer, token_stats = llm(prompt, model=model)

    relevance, rel_token_stats = evaluate_relevance(query, answer)

    t1 = time()
    took = t1 - t0

    answer_data = {
        "answer": answer,
        "model_used": model,
        "response_time": took,
        "relevance": relevance.get("Relevance", "UNKNOWN"),
        "relevance_explanation": relevance.get(
            "Explanation", "Failed to parse evaluation"
        ),
        "prompt_tokens": token_stats["prompt_tokens"],
        "completion_tokens": token_stats["completion_tokens"],
        "total_tokens": token_stats["total_tokens"],
        "eval_prompt_tokens": rel_token_stats["prompt_tokens"],
        "eval_completion_tokens": rel_token_stats["completion_tokens"],
        "eval_total_tokens": rel_token_stats["total_tokens"]
    }
    
    return answer_data
