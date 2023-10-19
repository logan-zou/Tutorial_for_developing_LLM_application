from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

import model_to_llm
import get_vectordb


class QA_chain_self():
    # 自定义 QA 链
    
    """"
    不带历史记录的问答链    
    """

    

    #基于召回结果和 query 结合起来构建的 prompt使用的默认提示模版
    default_template_rq = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
    案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
    {context}
    问题: {question}
    有用的回答:"""

    #基于大模型的问答 prompt 使用的默认提示模版
    #default_template_llm = """请回答下列问题:{question}"""
           
    

    def answer(self, model:str, question:str=None, template=default_template_rq, file_path:str=None, persist_path:str=None,api_key: str = None, embedding = "openai"):
        """"
        核心方法，调用问答链
        arguments: 
        - question：用户提问
        - model：调用的模型名称
        - top_k：返回检索的前k个相似文档
        - template：可以自定义提示模板，没有输入则使用默认的提示模板default_template_rq
        """

        if len(question) == 0:
            return ""
        
        vectordb = get_vectordb(file_path, persist_path, api_key, embedding)
        llm = model_to_llm(model)
        
        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                    template=template)
        # 自定义 QA 链
        qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                        retriever=vectordb.as_retriever(),
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})

        result = qa_chain({"query": question})
        return result["result"]   
