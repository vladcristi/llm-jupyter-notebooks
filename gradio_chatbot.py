import uuid

import gradio as gr
from model_utils import *
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_cohere import CohereEmbeddings
from langchain_openai import ChatOpenAI

from qdrant_client import QdrantClient, models

def clean_up_vector_db(QDRANT_URL):
    qdrant_client = QdrantClient(
            location=QDRANT_URL)
    
    existing_collections = qdrant_client.get_collections()
    existing_collections_name = [ entry.name for entry in existing_collections.collections ] 
    
    for collection in existing_collections.collections:
        qdrant_client.delete_collection(collection.name)

# Create one collection per each of the embedding models

def create_empty_collections(QDRANT_URL, embedding_models):
    qdrant_client = QdrantClient(
            location=QDRANT_URL)
    
    existing_collections = qdrant_client.get_collections()
    existing_collections_name = [ entry.name for entry in existing_collections.collections ] 

    for key, value in embedding_models.items():
        if key not in existing_collections_name:
            qdrant_client.create_collection(
                collection_name=key,
                vectors_config=models.VectorParams(size=value.get("size"), distance=models.Distance.COSINE),
            )

def fetch_session_hash(request: gr.Request):
    return request.session_hash


def setup_chatbot(llm_models, embedding_models, QDRANT_URL):
    with gr.Blocks(gr.themes.Soft(primary_hue=gr.themes.colors.slate, secondary_hue=gr.themes.colors.purple)) as demo:
        with gr.Row():
            with gr.Column(scale=0.5, variant='panel'):
                gr.Markdown("## RAG Conversation agent")
                instruction = gr.Textbox(label="System instruction", lines=3, value="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Keep the answer concise. {context}")
                
                session = gr.Textbox(value=uuid.uuid1, label="Session")
                gr.ChatInterface(
                    fn=bot,
                    additional_inputs=[
                        session,
                    ],
                )
                demo.load(fetch_session_hash, None, session)
    
            with gr.Column(scale=0.5, variant = 'panel'):
                gr.Markdown("## Select the Generation and Embedding Models")
                with gr.Row(equal_height=True):
    
                    with gr.Column():
                        available_llm_models = sorted([ entry for entry in llm_models ])
                        llm = gr.Dropdown(choices=available_llm_models,
                                          value=available_llm_models[0] if len(available_llm_models) else "",
                                          label="Select the LLM")
                        llm_api_key = gr.Textbox(label='Enter your valid LLM API KEY', type = "password")
                    
                        available_embedding_models = sorted([ entry for entry in embedding_models ])
                        embedding_model = gr.Dropdown(choices=available_embedding_models,
                                        value=available_embedding_models[0] if len(available_embedding_models) else "",
                                        label= "Select the embedding model")
                        embedding_api_key = gr.Textbox(label='Enter your embeddings API KEY', type = "password")
    
                    with gr.Column():
                        model_load_btn = gr.Button('Load model', variant='primary',scale=2)
                        load_success_msg = gr.Textbox(show_label=False,lines=1, placeholder="Model loading ...")
                
                gr.Markdown("## Upload Document")
                file = gr.File(type="filepath")
                with gr.Row(equal_height=True):
                    
    
                    with gr.Column(variant='compact'):
                        vector_index_btn = gr.Button('Create vector store', variant='primary', scale=1)
                        vector_index_msg_out = gr.Textbox(show_label=False, lines=1, scale=1, placeholder="Creating vectore store ...")
    
                vector_index_btn.click(lambda arg1, arg2, arg3, arg4: upload_and_create_vector_store(arg1, arg2, arg3, arg4, embedding_models, QDRANT_URL), [file, embedding_model, embedding_api_key, session], vector_index_msg_out)
                
                reset_inst_btn = gr.Button('Reset',variant='primary', size = 'sm')
    
                with gr.Accordion(label="Text generation tuning parameters"):
                    temperature = gr.Slider(label="temperature", minimum=0.1, maximum=1, value=0.7, step=0.05)
                    max_tokens = gr.Slider(label="max_tokens", minimum=500, maximum=8000, value=1000, step=1)
                    frequency_penalty = gr.Slider(label="frequency_penalty", minimum=0, maximum=2, value=0, step=0.1)
                    top_p=gr.Slider(label="top_p", minimum=0, maximum=1, value=0.9, step=0.05)
    
                model_load_btn.click(lambda arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10: load_model(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, embedding_models, llm_models, QDRANT_URL), [session, embedding_model, embedding_api_key, llm, llm_api_key, instruction, temperature, max_tokens, frequency_penalty, top_p], load_success_msg, api_name="load_model").success()
                reset_inst_btn.click(reset_sys_instruction, instruction, instruction)
            return demo
            

if __name__ == '__main__':

    embedding_models = {
        # "all-roberta-large-v1_1024d": {
        #     "model": "sentence-transformers/all-roberta-large-v1",
        #     "embedding_function": lambda api_key: HuggingFaceEmbeddings(
        #         model_name="sentence-transformers/all-roberta-large-v1"
        #     ),
        #     "size": 1024
        # },
        # "all-mpnet-base-v2_768d": {
        #     "model": "sentence-transformers/all-mpnet-base-v2",
        #     "embedding_function": lambda api_key: HuggingFaceEmbeddings(
        #         model_name="sentence-transformers/all-mpnet-base-v2"
        #     ),
        #     "size": 768
        # },
        # "cohere": {
        #     "model": "embed-english-light-v3.0",
        #     "embedding_size": 384,
        #     "embedding_function": lambda api_key: CohereEmbeddings(
        #         cohere_api_key=api_key,
        #         model="embed-english-light-v3.0"
        #     ),
        #     "size": 384
        # }
    }

    llm_models = {
        # "Meta-Llama-3-8B-Instruct": {
        #     "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        #     "llm_function": lambda api_key, kwargs={}: ChatOpenAI(
        #         base_url="http://llm/v1",
        #         model="meta-llama/Meta-Llama-3-8B-Instruct",
        #         api_key=api_key,
        #         **kwargs
        #     )
        # },
        # "nVidia NIM llama3-70b-instruct": {
        #     "model": "meta/llama3-70b-instruct",
        #     "llm_function": lambda api_key, kwargs={}: ChatOpenAI(
        #         base_url="https://integrate.api.nvidia.com/v1",
        #         model="meta/llama3-70b-instruct",
        #         api_key=api_key,
        #         **kwargs
        #     )
        # }
    }

    QDRANT_URL = "http://qdrant:6333"

    # clean_up_vector_db(QDRANT_URL)
    create_empty_collections(QDRANT_URL, embedding_models)

    demo = setup_chatbot(llm_models, embedding_models, QDRANT_URL)
    demo.queue(concurrency_count=3)
    demo.launch(debug=True, share=True)