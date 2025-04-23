from openai import AzureOpenAI
from dotenv import load_dotenv
import os
from azure.cosmos import CosmosClient

load_dotenv() # take environment variables from .env.

azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
azure_openai_embeddings_deployment = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
azure_openai_api_version = "2024-10-01-preview"
azure_openai_embedding_size = 1536

azure_cosmosdb_endpoint = os.getenv("AZURE_COSMOSDB_ENDPOINT")
azure_cosmosdb_key = os.getenv("AZURE_COSMOSDB_KEY")
azure_cosmosdb_database = "azureservicesdatabase01"
azure_cosmosdb_container = "azureservicescontainer01"

def hybrid_search(user_query, num_results):

    # Setup the connection
    cosmos_client = CosmosClient(url=azure_cosmosdb_endpoint, credential=azure_cosmosdb_key)
    database = cosmos_client.get_database_client(azure_cosmosdb_database)
    container = container = database.get_container_client(azure_cosmosdb_container)

    # Azure OpenAI client
    openai_client = AzureOpenAI(
    api_version=azure_openai_api_version,
    azure_endpoint=azure_openai_endpoint,
    azure_deployment=azure_openai_embeddings_deployment,
    api_key=azure_openai_key)

    response = openai_client.embeddings.create(input=user_query, 
                                               model=azure_openai_embeddings_deployment, 
                                               dimensions=1536)
    embedding = response.data[0].embedding


    # Build the query with str.format() method
    query = '''
        SELECT TOP {0} c.id, c.title, c.category, c.content
        FROM c
        ORDER BY RANK RRF 
            (VectorDistance(c.contentVector, {1}), FullTextScore(c.title, ['{2}']))
    '''.format(num_results, embedding, user_query)

    results = container.query_items(
            query=query,
            enable_cross_partition_query=True)

    items = [item for item in results]
    
    return items

def RAG_CosmosDb(user_query):
    
    # Azure OpenAI client
    openai_client = AzureOpenAI(
        api_version=azure_openai_api_version,
        azure_endpoint=azure_openai_endpoint,
        api_key=azure_openai_key)

    # Provide instructions to the model
    SYSTEM_PROMPT="""
    You are an AI assistant that helps users learn from the information found in the source material.
    Answer the query using only the sources provided below.
    Use bullets if the answer has multiple points.
    If the answer is longer than 3 sentences, provide a summary.
    Answer ONLY with the facts listed in the list of sources below. Cite your source when you answer the question
    If there isn't enough information below, say you don't know.
    Do not generate answers that don't use the sources below.
    Query: {query}
    Sources:\n{sources}
    """

    # User Query
    query = user_query

    results = hybrid_search(query, 5)

    # Use a unique separator to make the sources distinct. 
    # We chose repeated equal signs (=) followed by a newline because it's unlikely the source documents contain this sequence.
    sources_formatted = "=================\n".join([f"TITLE: {document['title']}, CONTENT: {document['content']}, CATEGORY: {document['category']}" for document in results])

    response = openai_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": SYSTEM_PROMPT.format(query=query, sources=sources_formatted)
            }
        ],
        model=azure_openai_deployment
    )

    print(response.choices[0].message.content)

if __name__ == "__main__":
    user_query = input("Enter your query: ")
    RAG_CosmosDb(user_query)
