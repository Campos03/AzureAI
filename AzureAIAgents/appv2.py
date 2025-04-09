import chainlit as cl
import json
from typing import Any, Callable, Set, Dict, List, Optional
import os
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.projects.models import FunctionTool, ToolSet, AzureAISearchTool, AzureAISearchQueryType


# Define the function to fetch weather information
def fetch_weather(location: str) -> str:
    """
    Fetches the weather information for the specified location.

    :param location (str): The location to fetch weather for.
    :return: Weather information as a JSON string.
    :rtype: str
    """
    # In a real-world scenario, you'd integrate with a weather API.
    # Here, we'll mock the response.
    mock_weather_data = {
        "New York": "Sunny, 25°C", 
        "London": "Cloudy, 18°C", 
        "Tokyo": "Rainy, 22°C"
    }
    weather = mock_weather_data.get(location, "Weather data not available for this location.")
    weather_json = json.dumps({"weather": weather})
    return weather_json

# Define the function to fetch restaurant information
def fetch_restaurant(location: str) -> str:
    """
    Fetches the restaurant information for the specified location.

    :param location (str): The location to fetch the restaurant for.
    :return: Restaurant information as a JSON string.
    :rtype: str
    """
    # In a real-world scenario, you'd integrate with a restaurant API.
    # Here, we'll mock the response.
    mock_restaurant_data = {
        "New York": "Tatiana by Kwame Onwuachi, Katz’s Delicatessen, Peter Luger Steakhouse, Sylvia's, Nathan's Famous", 
        "London": "St. JOHN, Señor Ceviche, Gloria and Circolo Popolare, Normah's, Bouchon Racine", 
        "Tokyo": "Chanko & Wanko Restaurant Asakusa Sumo Club, Sky Restaurant 634 Musashi, Ichiran, Shibuya, Rokkasen Otakibashiidori, Hakushu - Kobe Teppanyaki"
    }
    restaurant = mock_restaurant_data.get(location, "Restaurant data not available for this location.")
    restaurant_json = json.dumps({"restaurant": restaurant})
    return restaurant_json

# Define the function to fetch budget information
def fetch_budget() -> str:
    """
    Fetches the budget information for the specified location.
    :return: budget information as a JSON string.
    :rtype: str
    """
    # In a real-world scenario, you'd integrate with a another API.
    # Here, we'll mock the response.
    mock_budget_data = {
        "New York": """
            Budget Travelers: Around $121 per day. This includes staying in hostels, eating at budget restaurants, and using public transportation.
            Mid-Range Travelers: Approximately $324 per day. This covers mid-range hotels, dining at average restaurants, and some paid attractions.
            Luxury Travelers: About $923 per day. This includes luxury hotels, fine dining, and private transportation.
        """, 
        "London": """
            Budget Travelers: Around $75 per day. This includes staying in hostels, cooking your own meals, and using public transport.
            Mid-Range Travelers: Approximately $195 per day. This covers mid-range hotels, dining at average restaurants, and some paid attractions.
            Luxury Travelers: About $517 per day. This includes luxury hotels, fine dining, and private transportation.
        """, 
        "Tokyo": """
            Budget Travelers: Around $100 per day. This includes staying in hostels, eating at budget restaurants, and using public transportation.
            Mid-Range Travelers: Approximately $286 per day. This covers mid-range hotels, dining at average restaurants, and some paid attractions.
            Luxury Travelers: About $908 per day. This includes luxury hotels, fine dining, and private transportation.
        """
    }
    budget_json = json.dumps({"budget": mock_budget_data})
    return budget_json

def fetch_product_info(userquery: str) -> str:
    """
    Fetches the product information for the specified user query.
    :return: product information.
    :rtype: str
    """

    from azure.search.documents import SearchClient
    from azure.search.documents.models import VectorizableTextQuery
    from azure.core.credentials import AzureKeyCredential
    from openai import AzureOpenAI

    azure_search_service_admin_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
    azure_search_service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
    azure_search_service_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
    azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_openai_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

    # Get credential from Azure AI Search Admin key
    credential = AzureKeyCredential(azure_search_service_admin_key)
    search_client = SearchClient(endpoint=azure_search_service_endpoint, 
                                credential=credential, 
                                index_name=azure_search_service_index_name)

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
    query = userquery 

    # Convert query into vector form
    vector_query = VectorizableTextQuery(text=query, 
                                        k_nearest_neighbors=50, 
                                        fields="text_vector",
                                        weight=1)

    results = search_client.search(
        query_type="semantic", 
        semantic_configuration_name='my-semantic-config',
        search_text=query,
        vector_queries= [vector_query],
        select=["title","chunk"],
        top=5,
    )

    # Use a unique separator to make the sources distinct. 
    # We chose repeated equal signs (=) followed by a newline because it's unlikely the source documents contain this sequence.
    sources_formatted = "=================\n".join([f'TITLE: {document["title"]}, CONTENT: {document["chunk"]}' for document in results])

    response = openai_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": SYSTEM_PROMPT.format(query=query, sources=sources_formatted)
            }
        ],
        model=azure_openai_deployment
    )

    return response.choices[0].message.content


# Statically defined user functions for fast reference
user_functions: Set[Callable[..., Any]] = {
    fetch_weather, fetch_restaurant, fetch_budget, fetch_product_info
}

# Define the function to run the agent
def run_agent(user_input, project_client, thread, agent):  
    # Step 3: Add a message to the thread  
    message = project_client.agents.create_message(
        thread_id=thread.id,
        role="user",
        content=user_input,
    )
    print(f"Created message, ID: {message.id}")

    # Step 4 & 5: Create and process agent run in thread with tools
    run = project_client.agents.create_and_process_run(thread_id=thread.id, agent_id=agent.id)
    
    if run.status == "failed":
        print(f"Run failed: {run.last_error}")

    # Step 6: Display the Agent's Response
    elif run.status == 'completed':
            # Fetch all messages in the thread
            messages = project_client.agents.list_messages(thread_id=thread.id)
            if messages.data:
                agent_response = messages.data[0].content[0].text.value # Get the last assistant message
                print(f"Agent Response: {agent_response}") 
            else:
                print("No messages found.")
    
    return agent_response

# Define the function to delete the agent
def delete_agent(project_client, agent):
    # Delete the agent when done
    project_client.agents.delete_agent(agent.id)
    print("Deleted agent")
    

@cl.on_chat_start
def on_chat_start():
    
    global project_client
    global agent
    global thread

    project_connection_string = os.getenv("PROJECT_CONNECTION_STRING")
    # Create an Azure AI Client from a connection string, copied from your Azure AI Foundry project.    
    project_client = AIProjectClient.from_connection_string(
        credential=DefaultAzureCredential(),
        conn_str=project_connection_string,
    )

    # Initialize agent toolset with user functions
    functions = FunctionTool(user_functions)
    toolset = ToolSet()
    toolset.add(functions)

    # Create a new agent with the toolset
    agent = project_client.agents.create_agent(
        model="gpt-4o", 
        name="my-chainlit-agent", 
        instructions="""
            You are an AI Travel Agent. 
            You will answer questions about travel based on the tools provided.
            You have access to the following tools:
            - fetch_weather - fetches the weather information for a given location.
            - fetch_restaurant - fetches restaurant information for a given location.
            - fetch_budget - fetches budget information for a given location.
            - fetch_product_info - fetches product information such as travel insurance, luggage, wifi plan, accessories and other products based on user queries.
        """, 
        toolset=toolset,
    )

    print(f"Created agent, ID: {agent.id}")

    # Create a new thread for the agent
    thread = project_client.agents.create_thread()
    print(f"Created thread, ID: {thread.id}")
    
    print("A new chat session has started!")

@cl.on_message
async def main(message: cl.Message):

    # Call the agent with the user's message
    agent_response = run_agent(message.content, project_client, thread, agent)
    
    # Send a response back to the user
    await cl.Message(
        content=agent_response,
    ).send()

@cl.on_chat_end
def on_chat_end():
    # Delete the agent when done
    delete_agent(project_client, agent)
    print("The user disconnected!")

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Weather Queries",
            message="What is the weather like in New York?",
            icon="/public/weather.svg",
            ),

        cl.Starter(
            label="Restaurant Recommendations",
            message="What are some good restaurants in London?",
            icon="/public/food.svg",
            ),
        cl.Starter(
            label="Budget Information",
            message="How much does it cost to travel to Tokyo?",
            icon="/public/money.svg",
            ),
        cl.Starter(
            label="Budget Recommendations",
            message="If I have a budget of $300 for 4 days, where should I travel?",
            icon="/public/calculator.svg",
            ),
        cl.Starter(
            label="Suitcase Products",
            message="What suitcases do you have?",
            icon="/public/suitcase.svg",
            ),
        cl.Starter(
            label="Handcarry Bags",
            message="Do you have any bags available?",
            icon="/public/briefcase.svg",
            )
        ]