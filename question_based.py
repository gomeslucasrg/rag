# -*- encoding: utf-8 -*-
import os
import numpy as np
import math
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma, Redis
from langchain.chains import RetrievalQA, ConversationalRetrievalChain, MapReduceDocumentsChain, StuffDocumentsChain, \
    ReduceDocumentsChain
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from tenacity import retry, stop_after_attempt
from dotenv import load_dotenv

from variables import REDIS_ADDRESS, OPENAI_KEY

load_dotenv()

os.environ['OPENAI_API_KEY'] = OPENAI_KEY

import os, sys

def getnearpos(vector,value):
    idx = (np.abs(vector-value)).argmin()
    return idx

@retry(stop=stop_after_attempt(3))
def start(documentId):
    try:
        llm = ChatOpenAI(temperature=0.4, model_name="gpt-3.5-turbo-16k")
        vectorizer = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])

        rds = Redis.from_existing_index(
            vectorizer,
            redis_url=f"redis://{REDIS_ADDRESS}",
            index_name=documentId
        )

        retriever = rds.as_retriever(search_type="similarity", search_kwargs={"k": 6})

        template = """You are a representative and should keep your responses based on the context to answer the question.
        If there is no context-related data for the question, simply say 'I don't have that answer,' do not attempt to invent an answer or
        look outside the context. Use a maximum of three sentences and keep the response as concise as possible. At the end of the response,
        insert points that encourage the conversation to move towards a sales closure if you believe the question allows for it.
      
        {context}
      
        Question: {question}
        Helpful Answer in portuguese:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        PROMPT = PromptTemplate(
            template=template, input_variables=["context", "question"]
        )
        chain_type_kwargs = {"prompt": PROMPT}

        qa_chain = RetrievalQA.from_chain_type(
            llm,
            chain_type="stuff",
            retriever= retriever,
            chain_type_kwargs= chain_type_kwargs
        )
        question = "Essa aula fala sobre o que ?"

        #result = qa_chain({"question": question})
        result = qa_chain.run(question)

        print(result)

        return None
    except Exception as error:
        print(error)
        print(
            type(error).__name__,  # TypeError
            __file__,  # /tmp/example.py
            error.__traceback__.tb_lineno  # 2
        )

def isPreviousOrNext(cluster_index_list, index):

    for i in cluster_index_list:
        previous = i - 1
        next = i + 1

        if next == index or previous == index:
            return True


    return False

if __name__ == '__main__':
    start('document-01')


