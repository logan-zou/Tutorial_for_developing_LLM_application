# 导入必要的库
import os                # 用于操作系统相关的操作，例如读取环境变量
import io                # 用于处理流式数据（例如文件流）
import IPython.display   # 用于在 IPython 环境中显示数据，例如图片
import requests          # 用于进行 HTTP 请求，例如 GET 和 POST 请求
import zhipuai
import sys

sys.path.append('..')

sys.path.append('../..')
# import llm.ZhipuAILLM
from llm.zhipuai_llm import ZhipuAILLM
from embedding import ZhipuAIEmbeddings
import gradio as gr
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from langchain.llms import OpenAI
from langchain.llms import HuggingFacePipeline
# 设置请求的默认超时时间为60秒
requests.adapters.DEFAULT_TIMEOUT = 60
persist_directory = "notebook/C7前后端搭建/docs/chroma/knowledge_base"
# 导入 dotenv 库的函数
# dotenv 允许您从 .env 文件中读取环境变量
# 这在开发时特别有用，可以避免将敏感信息（如API密钥）硬编码到代码中
from dotenv import load_dotenv, find_dotenv

# 寻找 .env 文件并加载它的内容
# 这允许您使用 os.environ 来读取在 .env 文件中设置的环境变量
_ = load_dotenv(find_dotenv())
llm_model_list = ['zhipuai', 'chatgpt', 'wenxin']
init_llm = "zhipuai"
init_embedding_model = "zhipuai"

block = gr.Blocks()
with block as demo:
    gr.Markdown("""<h1><center>Chat Robot</center></h1>
    <center>Local Knowledge Base Q&A with llm</center>
    """)
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=480) 
            # 创建一个文本框组件，用于输入 prompt。
            msg = gr.Textbox(label="Prompt/问题")

            with gr.Row():
                # 创建一个提交按钮。
                db_with_his_btn = gr.Button("Chat db with history")
                db_wo_his_btn = gr.Button("Chat db without history")
                llm_btn = gr.Button("Chat with llm")
            with gr.Row():
                # 创建一个清除按钮，用于清除文本框和聊天机器人组件的内容。
                clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

        with gr.Column(scale=1):
            file = gr.File(label='请上传知识库文件',file_count='directory',
                file_types=['.txt', '.md', '.docx', '.pdf'])
            with gr.Row():
                init_db = gr.Button("知识库文件向量化")
            model_argument = gr.Accordion("参数配置", open=False)
            with model_argument:
                top_k = gr.Slider(1,
                                10,
                                value=3,
                                step=1,
                                label="vector db search top k",
                                interactive=True)

                history_len = gr.Slider(0,
                                        5,
                                        value=3,
                                        step=1,
                                        label="history length",
                                        interactive=True)

                temperature = gr.Slider(0,
                                        1,
                                        value=0.00,
                                        step=0.01,
                                        label="llm temperature",
                                        interactive=True)
            model_select = gr.Accordion("模型选择")
            with model_select:
                llm = gr.Dropdown(
                    llm_model_list,
                    label="large language model",
                    value=init_llm,
                    interactive=True)

                embeddings = gr.Dropdown(llm_model_list,
                                                label="Embedding model",
                                                value=init_embedding_model)

        # 设置按钮的点击事件。当点击时，调用上面定义的 chat_with_db 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        init_db.click(
            init_knowledge_db,
            show_progress=True,
            inputs=[file],
            outputs=[],
        )
        
        # 设置按钮的点击事件。当点击时，调用上面定义的 chat_with_db 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        db_with_his_btn.click(Chat_QA_chain_self.answer, inputs=[msg, chatbot,  llm, embeddings, history_len, top_k, temperature], outputs=[msg, chatbot])
        db_wo_his_btn.click(QA_chain_self.answer, inputs=[msg, chatbot, llm, embeddings, top_k, temperature], outputs=[msg, chatbot])
        llm_btn.click(get_completion, inputs=[msg, chatbot, llm, history_len, top_k, temperature], outputs=[msg, chatbot])

        # 设置文本框的提交事件（即按下Enter键时）。功能与上面的按钮点击事件相同。
        msg.submit(get_completion, inputs=[msg, chatbot,  llm, embeddings, history_len, top_k, temperature], outputs=[msg, chatbot]) 
        # 点击后清空后端存储的聊天记录
        # clear.click(clear_history)
    gr.Markdown("""提醒：<br>
    1. 使用时请先上传自己的知识文件，并且文件中不含某些特殊字符，否则将返回error. <br>
    """)
# threads to consume the request
gr.close_all()
# 启动新的 Gradio 应用，设置分享功能为 True，并使用环境变量 PORT1 指定服务器端口。
# demo.launch(share=True, server_port=int(os.environ['PORT1']))
# 直接启动
demo.launch()