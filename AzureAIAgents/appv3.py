## az login is needed before running this code

import os
import chainlit as cl
import logging
from dotenv import load_dotenv
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

# Load environment variables
load_dotenv()

# Disable verbose connection logs
logger = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
logger.setLevel(logging.WARNING)

# Load environment variables
AIPROJECT_CONNECTION_STRING = os.getenv("PROJECT_CONNECTION_STRING")
AGENT_ID = os.getenv("AGENT_ID")

# Create an instance of the AIProjectClient using DefaultAzureCredential
project_client = AIProjectClient.from_connection_string(
	conn_str=AIPROJECT_CONNECTION_STRING, credential=DefaultAzureCredential()
)

# Define the function to run the agent
def run_agent(user_input, project_client, thread_id):  

    print("Entering run agent...")
    print(AGENT_ID)
    print(AIPROJECT_CONNECTION_STRING)

    # Add a message to the thread  
    message = project_client.agents.create_message(
        thread_id=thread_id,
        role="user",
        content=user_input,
    )
    print(f"Created message, ID: {message.id}")

    # Create and process agent run in thread with tools
    run = project_client.agents.create_and_process_run(thread_id=thread_id, agent_id=AGENT_ID)
    
    if run.status == "failed":
        print(f"Run failed: {run.last_error}")

    # Display the Agent's Response
    elif run.status == 'completed':
            # Fetch all messages in the thread
            messages = project_client.agents.list_messages(thread_id=thread_id)
            if messages.data:
                agent_response = messages.data[0]  # Get the last assistant message
                print(f"Agent Response: {agent_response.content[0].text.value}") 
            else:
                print("No messages found.")
    
    return agent_response

@cl.on_chat_start
async def on_chat_start():
    print(AGENT_ID)
    print(AIPROJECT_CONNECTION_STRING)
    if not cl.user_session.get("thread_id"):
	    
        thread = project_client.agents.create_thread()
        cl.user_session.set("thread_id", thread.id)
        print(f"New Thread ID: {thread.id}")

@cl.on_message
async def main(message: cl.Message):
    thread_id = cl.user_session.get("thread_id")

    # Call the agent with the user's message
    agent_response = run_agent(message.content, project_client, thread_id)
    
    # Send a response back to the user
    await cl.Message(
        content=agent_response,
    ).send()
    
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