import os
import pandas
from tqdm.auto import tqdm
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers
from db import init_db


os.environ['RUN_TIMEZONE_CHECK'] = '0'


load_dotenv()


ELASTIC_URL = os.getenv("ELASTIC_URL_LOCAL", "http://localhost:9200")
INDEX_NAME = os.getenv("INDEX_NAME", "movies")

len_documents = 722359


def load_data(data_path='../data/movies.csv'):
    df = pandas.read_csv(data_path, keep_default_na=False)
    df['release_date'].replace('', '1970-01-01', inplace=True)

    documents = df.to_dict(orient='records')
    return documents


def setup_elasticsearch(es_client):
    print("Setting up Elasticsearch...")
    

    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "id": {"type": "integer", "null_value": 0},
                "title": {"type": "text"},
                "genres": {"type": "text"},
                "original_language": {"type": "keyword"},
                "overview": {"type": "text"},
                "popularity": {"type": "float"},
                "production_companies": {"type": "text"},
                "release_date": {"type": "date", "format": "yyyy-MM-dd", "null_value": "1970-01-01"},
                "budget": {"type": "float"},
                "revenue": {"type": "float"},
                "runtime": {"type": "float"},
                "status": {"type": "keyword"},
                "tagline": {"type": "text"},
                "vote_average": {"type": "float"},
                "vote_count": {"type": "float"},
                "credits": {"type": "text"},
                "keywords": {"type": "text"},
            }
        }
    }

    es_client.indices.create(index=INDEX_NAME, body=index_settings)
    print(f"Elasticsearch index '{INDEX_NAME}' created")


# Function to create the bulk actions
def generate_actions(documents):
    for doc in documents:
        yield {
            "_index": INDEX_NAME,
            "_source": doc
        }


# Bulk indexing function with progress bar
def bulk_index(es_client, documents, batch_size=2000):
    total_documents = len(documents)
    progress_bar = tqdm(total=total_documents, desc="Indexing Progress")

    for success, info in helpers.parallel_bulk(
        es_client,
        generate_actions(documents),
        chunk_size=batch_size
    ):
        if not success:
            print('A document failed:', info)
        progress_bar.update(batch_size)

    progress_bar.close()


def main():
    
    print("Initializing database...")
    init_db()
    
    es_client = Elasticsearch(ELASTIC_URL, request_timeout=40)
    
    if es_client.indices.exists(index=INDEX_NAME) and es_client.count(index=INDEX_NAME).get('count') == len_documents:
        print(f"Index '{INDEX_NAME}' already exists with {len_documents} documents.")
        return
    else:
        try:
            es_client.indices.delete(index=INDEX_NAME)
        except Exception as e:
            print(f"Error deleting index: {e}")
    
    print("Starting the indexing process...")
    documents = load_data()
    setup_elasticsearch(es_client)
    bulk_index(es_client, documents)

    print("Indexing process completed successfully!")

if __name__ == "__main__":
    main()