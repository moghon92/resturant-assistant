
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import LLMMathChain
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, Tool, tool, initialize_agent
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent, create_retriever_tool
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import MessagesPlaceholder
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage, AIMessage, HumanMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor


import logging
import sys
import os
from pprint import pprint
import json


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

os.environ["OPENAI_API_KEY"] = "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"
Global_order = []

def pre_process_menu():
    with open('menu.json', 'r') as file:
        menu = json.load(file)
        
    menu_items = []

    counter = 0
    for i in range(len(menu)):
        section = menu[i]['sections'][0] #???
        for j in range(len(section['categories'])):
            category = section['categories'][j]
            for k in range(len(category['items'])):
                item = category['items'][k]
                #print(item)
                if 'name' in item.keys() and 'desc' in item.keys() and 'price' in item.keys():
                    counter+=1
                    item_keys = (f"({counter})\n"
                     f"name: {item['name']}\n"
                     f"desc: {item['desc']}\n"
                     f"price: {float(item['price'])}\n"
                     f"category: {category['name']}\n"
                     f"section:  {menu[i]['name']}\n"
                     f"type: {section['name']}\n"
                    )
                    #print(item_keys)

                    menu_items.append(item_keys)
                    
    return menu_items

def get_retriever(menu_items):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", r"(?<=\. )", " ", ""],
        chunk_size=1024,
        chunk_overlap=250
    )

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts(menu_items, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 10})
    
    return retriever



def define_tools(retriever, menu_items, llm):
    
    @tool
    def sample_menu():
        """Used to give recommendations to the customer"""
        import numpy as np
        #idx= np.random.randint(0,len(menu_items)-1,5)
        return np.array(menu_items)[:5].tolist()

    @tool
    def save_order(order_item:str):
        """takes as input one item the customer ordered and saves it in a list. use this tool whenver the customer orders as item."""
        Global_order.append(order_item)

    @tool
    def get_orders_so_far():
        """gives back a summary of all items ordered so far."""
        return Global_order
    
    
    ret_tool = create_retriever_tool(
    retriever,
    "search_menu",
    "useful for when you want to answer questions about the menu. The input to this tool should be a complete english sentence.",
    )

    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

    calc_tool = Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math (add, subtract,..",
    )

    return [ret_tool, calc_tool, sample_menu, save_order, get_orders_so_far]


def setup_agent(tools, llm):
    memory_key = "history"    
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                 "You are a helpful resturant waiter. Your role is to help guide the guest to complete an order. "
                "Feel free to use any tools available to look up. "
                "Give short answers and be super friendly and funny. Whenver possible use sample_menu tool to give menu recommendations."
                "Whenver the guest orders an item, save it using the save_order tool and then ask  the guet 'what else can i get for you?'. "
                "If the guest doesn't want anythng else use the get_orders_so_far tool to summarize the guest order and then tell the guest the total price (in AED) of thier order." ,
            ),
            MessagesPlaceholder(variable_name=memory_key),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    
    llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            ),
            "history": lambda x: x["history"],
        }
        | prompt
        | llm_with_tools
        | OpenAIFunctionsAgentOutputParser()
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    
    return agent_executor


def run_agent(user_msg, chat_history, stream_handler):
    llm = ChatOpenAI(
        model='gpt-3.5-turbo',
        temperature=0.0,
        streaming=True,
        request_timeout=10,
        callbacks=[stream_handler]
    )
    
    menu_items = pre_process_menu()
    retriever = get_retriever(menu_items)
    tools = define_tools(retriever, menu_items, llm)
    agent_executor = setup_agent(tools, llm)

    result = agent_executor.invoke({"input": user_msg, "history": chat_history})
    chat_history.extend(
        [
            HumanMessage(content=user_msg),
            AIMessage(content=result["output"]),
        ]
    )

    return {
        'output': result["output"],
        'updated_chat_history': chat_history
        }
        

    
    
