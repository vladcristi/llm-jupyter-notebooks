import os, shutil

from model_index import *
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.prompts import PromptTemplate, ChatPromptTemplate

# Global environment variables used to store state based on session_id
store = {}
conversation_rag_chain = {}

def load_model(session_id, embedding_model, embedding_api_key, llm_model, llm_api_key, instruction, temperature, max_tokens, frequency_penalty, top_p, embedding_models, llm_models, QDRANT_URL):

    global conversation_rag_chain

    # Setting up embedding
    
    embedding_function = embedding_models.get(embedding_model, {}).get("embedding_function", None)
    if not embedding_function:
        return "Selected embedding model is not defined."

    embedding = embedding_function(api_key=embedding_api_key)

    # Setting up LLM

    llm_function = llm_models.get(llm_model, {}).get("llm_function", None)
    
    if not llm_function:
        return "Selected llm model has not function defined."

    llm = llm_function(
        api_key=llm_api_key, 
        kwargs = {
            'temperature': temperature,
            'max_tokens': max_tokens,
            'frequency_penalty': frequency_penalty,
            'top_p': top_p
        }
    )
    
    # Setting up connection to Qdrant VectorDB
    qdrant_client = QdrantClient(
        location=QDRANT_URL)
    
    vectordb = Qdrant(
        client=qdrant_client,
        collection_name=embedding_model,
        embeddings=embedding
    )
    
    retriever = vectordb.as_retriever()

    ### Contextualize question ###
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    ### Answer question ###
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", instruction),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    document_prompt = PromptTemplate(
		            input_variables=["page_content", "source"], 
		            template="Context:\n{page_content}\nSource:{source}"
		        )
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt, document_prompt=document_prompt)#, output_parser=CustomOutputParser())
    
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    
    conversation_rag_chain[session_id] = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    
    if conversation_rag_chain.get(session_id, None) is None:
        return f"Failed to load the model {llm_model}"
        
    return f"'{llm_model}' and '{embedding_model}' models loaded"


def upload_and_create_vector_store(file, embedding_model, embedding_api_key, session, embedding_models, qdrant_url):
    
    # Save the uploaded file to a permanent location
    if not file:
        return "File not ready or no file selected."
        
    split_file_name = file.split("/")
    file_name = split_file_name[-1]

    data_folder = os.path.join(os.getcwd(), "data")
    os.makedirs(data_folder, exist_ok=True)
    permanent_file_path = os.path.join(data_folder, file_name)
    shutil.copy(file.name, permanent_file_path)

    # Access the path of the saved file
    print(f"File saved to: {permanent_file_path}")

    
    embedding_function = embedding_models.get(embedding_model, {}).get("embedding_function", None)
    
    if not embedding_function:
        return "Selected model not supported."

    embedding = embedding_function(api_key=embedding_api_key)
    
    collection_name = embedding_model
    index_success_msg = create_vector_store_index(qdrant_url, collection_name, permanent_file_path, embedding)
    
    return index_success_msg

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def reset_sys_instruction(instruction):

    default_inst = "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Keep the answer concise. {context}"
    return default_inst


def bot(question, history, session_id):
    global conversation_rag_chain
    
    result = conversation_rag_chain[session_id].invoke({"input": question},
        config={
            "configurable": {"session_id": session_id}
        },
    )
    
    return result["answer"]