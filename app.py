import chainlit as cl
from langchain import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.embeddings import SentenceTransformerEmbeddings
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
import os
import json
from langchain.llms import HuggingFacePipeline
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from transformers import (
    GenerationConfig,
    pipeline,
)

# from load_models import (
#     load_quantized_model_awq,
#     load_quantized_model_gguf_ggml,
#     load_quantized_model_qptq,
#     load_full_model,
# )

# from constants import (
#     EMBEDDING_MODEL_NAME,
#     PERSIST_DIRECTORY,
#     MODEL_ID,
#     MODEL_BASENAME,
#     MAX_NEW_TOKENS,
#     MODELS_PATH,
#     CHROMA_SETTINGS
# )
from langchain.llms import Ollama
# app = FastAPI()
# templates = Jinja2Templates(directory="templates")
# app.mount("/static", StaticFiles(directory="static"), name="static")
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.base import Runnable
print("static folder mounted")

# # local_llm = "meditron-7b.Q4_K_M.gguf"
# local_llm = "meditron-7b.Q2_K.gguf"

# config = {
# 'max_new_tokens': 512,
# 'context_length': 1024,
# 'repetition_penalty': 1.1,
# 'temperature': 0.1,
# 'top_k': 50,
# 'top_p': 0.9,
# 'stream': True,
# 'threads': int(os.cpu_count() / 2)
# }
# try:
#    llm = CTransformers(
#    model='TheBloke/meditron-7B-GGUF',
#    )
#    # llm = CTransformers(
#    # model=local_llm,
#    # model_type="llama",
#    # lib="avx",
#    # **config
#    # )
# except Exception as e:
#    print(e)
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline , TextIteratorStreamer

# model_name_or_path = "TheBloke/meditron-7B-GPTQ"
# # To use a different branch, change revision
# # For example: revision="gptq-4bit-32g-actorder_True"
# model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
#                                              device_map="auto",
#                                              trust_remote_code=False,
#                                              revision="main")

# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
# generation_config = GenerationConfig.from_pretrained(model_name_or_path)
# streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_length=MAX_NEW_TOKENS,
#     temperature=0.2,
#     # top_p=0.95,
#     repetition_penalty=1.15,
#     streamer=streamer,
#     generation_config=generation_config,
# )


# llm = HuggingFacePipeline(pipeline=pipe,
#                          # device='auto',  # replace with device_map="auto" to use the accelerate library.
#     pipeline_kwargs={"max_new_tokens": 10, "max_length":10}
# )
# llm = llm.bind(stop=["\n\n"])

# callbacks = [StreamingStdOutCallbackHandler()]
# callbacks = []
# llm = load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging)
# llm = Ollama(model='meditron',verbose=True, callbacks=callbacks)

# import openai, os
# from langchain.llms import OpenAI
# openai.api_key = "EMPTY"
# openai.api_base = "http://34.90.142.9:8000/v1"
# os.environ['OPENAI_API_KEY'] = openai.api_key
# llm = OpenAI(model="TheBloke/meditron-7B-AWQ")
from langchain.llms import LlamaCpp
llm = LlamaCpp(
    model_path="meditron-7b-chat.Q4_K_M.gguf",
    temperature=0.1,
    max_tokens=1024,
    top_p=1,
    stop = ["\n\n"],
    # callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)
# from langchain.chat_models import ChatLiteLLM
# llm = ChatLiteLLM(
#      model="ollama/zephyr", 
#     api_base="http://localhost:11434"
# )
print(llm, "LLM Initialized....")

prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

url = "http://localhost:6333"

client = QdrantClient(
    url=url, prefer_grpc=False
)

db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_db1")

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

retriever = db.as_retriever(search_kwargs={"k":1})

@cl.on_chat_start
async def start():
    print(prompt)
    chain_type_kwargs = {"prompt": prompt}
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs, verbose=True)
    msg = cl.Message(content="Firing up the research info bot...")
    await msg.send()
    msg.content = "Hi, welcome to research info bot. What is your query?"
    await msg.update()
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    # msg = cl.Message(content="")

    # async for chunk in chain.astream(message,
    #     config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    # ):
    #     await msg.stream_token(chunk)

    # await msg.send()
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    # res=await chain.acall(message, callbacks=[cb])
    res = await chain.acall(message.content, callbacks=[cb])
    print(f"response: {res}")
    answer = res["result"]
    answer = answer.replace(".", ".\n")
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources: " + str(str(sources))
    else:
        answer += f"\nNo Sources found"

    await cl.Message(content=answer).send()

    
# @cl.on_message
# async def main(message: str):

# #     response = qa(query)
# #     print(response)
# #     answer = response['result']
# #     source_document = response['source_documents'][0].page_content
# #     doc = response['source_documents'][0].metadata['source']
# #     response_data = jsonable_encoder(json.dumps({"answer": answer, "source_document": source_document, "doc": doc}))
    
# #     res = Response(response_data)
#     response = await cl.make_async(qa)(message)
#     sentences = response['answers'][0].answer.split('\n')

#     # Check if the last sentence doesn't end with '.', '?', or '!'
#     if sentences and not sentences[-1].strip().endswith(('.', '?', '!')):
#         # Remove the last sentence
#         sentences.pop()

#     result = '\n'.join(sentences[1:])
#     await cl.Message(author="Bot", content=result).send()
    
# @app.get("/", response_class=HTMLResponse)
# async def read_root(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# @app.post("/get_response")
# async def get_response(query: str = Form(...)):
#     chain_type_kwargs = {"prompt": prompt}
#     qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs, verbose=True)
#     response = qa(query)
#     print(response)
#     answer = response['result']
#     source_document = response['source_documents'][0].page_content
#     doc = response['source_documents'][0].metadata['source']
#     response_data = jsonable_encoder(json.dumps({"answer": answer, "source_document": source_document, "doc": doc}))
    
#     res = Response(response_data)
#     return res
