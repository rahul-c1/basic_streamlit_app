
# RETRIEVAL-AUGMENTED GENERATION (RAG)
# ***


# streamlit run path_to_app.py

"""
What are the top 3 things needed to do principal component analysis (pca)?
"""

from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

import streamlit as st
import yaml
import uuid


# Initialize the Streamlit app
st.set_page_config(page_title="Drivebot", layout="wide")
st.title("Drivebot")

import pandas as pd
import yaml
from pprint import pprint
import boto3
import json
from botocore.exceptions import NoCredentialsError, PartialCredentialsError


import os
from dotenv import load_dotenv
# Load the .env file
load_dotenv()

# Function to read and print all environment variables
def read_env_variables():
    print("Environment variables:")
    for key, value in os.environ.items():
        print(f"{key}: {value}")


# Function to get a specific environment variable
def get_env_variable(key):
    value = os.getenv(key)
    if value is not None:
        #print(f"{key}: {value}")
        return value
    else:
        print(f"Environment variable '{key}' not found")

aws_access_key_id = get_env_variable("aws_access_key_id")
aws_secret_access_keys = get_env_variable("aws_secret_access_keys")

pdf_path = "C://Users/rasharma/OneDrive - Santander Office 365/Documents/CRM/Ad-hoc Request/Pending/Drivebot WatsonX/data/"


# Configuration
region_name = 'us-west-2'
session = boto3.Session(aws_access_key_id=aws_access_key_id,aws_secret_access_key=aws_secret_access_keys , region_name='us-west-2')
bedrock = session.client(service_name='bedrock-runtime',verify=False)

# Set up Chat Memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

view_messages = st.expander("View the message contents in session state")
from langchain_community.chat_models import BedrockChat

# Function to create a Bedrock LangChain client.
def get_llm():
    try:
        #session = boto3.Session(profile_name=profile_name, region_name=region_name) 
        #bedrock = session.client(service_name='bedrock-runtime', region_name=region_name) 

        model_kwargs = {
            #"max_tokens": 512,
            "temperature": 0,
            "maxTokenCount": 4096,
            "stopSequences": ["User:"],
            "temperature": 0,
            "topP": 0.9
            #"top_k": 250,
            #"top_p": 1,
            #"stop_sequences": ["\n\nHuman:"]
        }
        llm = BedrockChat(
            model_id="amazon.titan-text-express-v1",
            model_kwargs=model_kwargs,
            client=bedrock
            #credentials_profile_name=profile_name,
            #region_name=region_name
        )
        return llm
    except NoCredentialsError:
        print("No AWS credentials found. Please configure your AWS credentials.")
    except PartialCredentialsError:
        print("Incomplete AWS credentials found. Please check your AWS credentials.")
    except Exception as e:
        print(f"An error occurred while creating the Bedrock client: {e}")



def create_rag_chain():
    
    from langchain_community.embeddings import BedrockEmbeddings

    bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)


    vectorstore = Chroma(
        persist_directory=(os.path.join(pdf_path+"chroma_index")),
        embedding_function=bedrock_embeddings
    )
    
    retriever = vectorstore.as_retriever()
    
    llm = get_llm()


    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # * 2. Answer question based on Chat Context
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # * Combine both RAG + Chat Message History
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: msgs,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

rag_chain = create_rag_chain()

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if question := st.chat_input("Enter your  question here:", key="query_input"):
    with st.spinner("Thinking..."):
        st.chat_message("human").write(question)     
           
        response = rag_chain.invoke(
            {"input": question}, 
            config={
                "configurable": {"session_id": "any"}
            },
        )
        # Debug response
        # print(response)
        # print("\n")
  
        st.chat_message("ai").write(response['answer'])

# * NEW: View the messages for debugging
# Draw the messages at the end, so newly generated ones show up immediately
with view_messages:
    """
    Message History initialized with:
    ```python
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    ```

    Contents of `st.session_state.langchain_messages`:
    """
    view_messages.json(st.session_state.langchain_messages)
