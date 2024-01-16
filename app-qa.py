import os
from typing import List
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.docstore.document import Document
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import PromptTemplate
from langchain.tools import DuckDuckGoSearchRun
import chainlit as cl
from chainlit.types import AskFileResponse
from langchain.chat_models import ChatLiteLLM
import litellm
from langchain.document_transformers import (
    EmbeddingsClusteringFilter,
    EmbeddingsRedundantFilter,
)
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.vectorstores import Chroma
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.retrievers import PubMedRetriever
litellm.set_verbose=True
# chat = ChatLiteLLM(model="gpt-3.5-turbo")
chat = ChatLiteLLM(
    model="ollama/meditron-chat",
    # model="ollama/meditron-normal",
    streaming=True,
    verbose=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        api_base="http://localhost:11434",
    max_tokens=256,
    temperature=0.0
)
index_name = "langchain-demo"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# embeddings = OpenAIEmbeddings()
embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

namespaces = set()

from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
url = "http://localhost:6333"

client = QdrantClient(
    url=url, prefer_grpc=False
)

db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_db1")
db1 = Qdrant(client=client, embeddings=embeddings, collection_name="db-2048")
search = DuckDuckGoSearchRun()
os.environ['GOOGLE_API_KEY'] = '83d8c79d30746834c392ff4a3607b38c4c98b354'
os.environ['GOOGLE_CSE_ID'] = 'e0aa612f7131a4ea0'

search = GoogleSearchAPIWrapper()
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k":3})
retriever1 = db1.as_retriever(search_kwargs={"k":3})
# Vectorstore
vectorstore = Chroma(
    embedding_function=embeddings, persist_directory="./chroma_db_oai"
)

web_research_retriever = WebResearchRetriever.from_llm(
    vectorstore=vectorstore, llm=chat, search=search
)
pubmed = PubMedRetriever()
lotr = MergerRetriever(retrievers=[ retriever, retriever1, pubmed])
filter = EmbeddingsRedundantFilter(embeddings=embeddings)
pipeline = DocumentCompressorPipeline(transformers=[filter])
compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline, base_retriever=lotr
)
from langchain.document_transformers import LongContextReorder

filter = EmbeddingsRedundantFilter(embeddings=embeddings)
reordering = LongContextReorder()
pipeline = DocumentCompressorPipeline(transformers=[filter, reordering])
compression_retriever_reordered = ContextualCompressionRetriever(
    base_compressor=pipeline, base_retriever=lotr
)

prompt_template1 = "Use the following pieces of context and chat history to answer the question at the end.\n" + "If you don't know the answer, just say that you don't know, " + "don't try to make up an answer.\n\n" + "{context}\n\nChat history: {chat_history}\n\nQuestion: {question}  \nHelpful Answer:"
prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
# prompt = PromptTemplate(template=prompt_template1, input_variables=['context', 'question', 'chat_history'])

@cl.on_chat_start
async def start():
    # await cl.Avatar(
    #     name="Chatbot",
    #     url="https://avatars.githubusercontent.com/u/128686189?s=400&u=a1d1553023f8ea0921fba0debbe92a8c5f840dd9&v=4",
    # ).send()
    # files = None
    # while files is None:
    #     files = await cl.AskFileMessage(
    #         content=welcome_message,
    #         accept=["text/plain", "application/pdf"],
    #         max_size_mb=20,
    #         timeout=180,
    #         disable_human_feedback=True,
    #     ).send()

    # file = files[0]

    # msg = cl.Message(
    #     content=f"Processing `{file.name}`...", disable_human_feedback=True
    # )
    # await msg.send()

    # # No async implementation in the Pinecone client, fallback to sync
    # docsearch = await cl.make_async(get_docsearch)(file)
    chain_type_kwargs = {"prompt": prompt}
    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        # ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True),
        chat,
        chain_type="stuff",
        retriever=compression_retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs=chain_type_kwargs,
        rephrase_question= False
    )

    # Let the user know that the system is ready
    # msg.content = f"`{file.name}` processed. You can now ask questions!"
    # await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()