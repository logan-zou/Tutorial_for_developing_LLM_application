# 首先实现基本配置
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.document_loaders import UnstructuredFileLoader
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# from langchain.llms import OpenAI
# from langchain.llms import HuggingFacePipeline
from  zhipuai_embedding import ZhipuAIEmbeddings
import gradio as gr
# 使用前配置自己的 api 到环境变量中如
import os
import openai
import sys

from dotenv import load_dotenv, find_dotenv

# _ = load_dotenv(find_dotenv()) # read local .env fileopenai.api_key  = os.environ['OPENAI_API_KEY']
# openai.api_key  = os.environ['OPENAI_API_KEY']
db_path = "../../knowledge_base"
persist_directory = "../../vector_data_base"
def file_loader(file, loaders):
    if type(file) == gr.File:
        file = file.name
    if not os.path.isfile(file):
        [file_loader(file, loaders) for file in  os.listdir(file)]
        return

    file_type = file.split('.')[-1]
    if file_type == 'pdf':
        loader.append(PyMuPDFLoader(file))
    elif file_type == 'md':
        loader.append(UnstructuredMarkdownLoader(file))
    elif file_type == 'txt':
        loader.append(UnstructuredFileLoader(file))
    else:
        loader = None
    return

def create_db(files, embeddings):
    """
    该函数用于加载 PDF 文件，切分文档，生成文档的嵌入向量，创建向量数据库。

    参数:
    file: 存放文件的路径。
    embeddings: 用于生产 Embedding 的模型

    返回:
    vectordb: 创建的数据库。
    """
    if len(files) == 0:
        return "can't load empty file"
    if len(files) == 1:
        files = [files]
    loaders = []
    [file_loader(file, loaders)  for file in files]
    docs = []
    for loader in loaders:
        if loader is not None:
            docs.extend(loader.load())

    # 切分文档
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    split_docs = text_splitter.split_documents(docs[:10])

    # 定义持久化路径
    persist_directory = '../knowledge_base/chroma'

    # 加载数据库
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
    )

def presit_knowledge_db(vectordb):
    """
    该函数用于持久化向量数据库。

    参数:
    vectordb: 要持久化的向量数据库。
    """
    vectordb.persist()

def load_knowledge_db(path, embeddings):
    """
    该函数用于加载向量数据库。

    参数:
    path: 要加载的向量数据库路径。
    embeddings: 向量数据库使用的 embedding 模型。

    返回:
    vectordb: 加载的数据库。
    """
    vectordb = Chroma.from_documents(
        embeddings=embeddings,
        persist_directory=path.name  
    )
    return vectordb

if __name__ == "__main__":
    create_db(db_path, ZhipuAIEmbeddings)