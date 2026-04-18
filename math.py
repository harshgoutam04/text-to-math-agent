from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_classic.chains import LLMMathChain, LLMChain
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents import Tool, initialize_agent
from langchain_core.prompts import PromptTemplate
from langchain_community.callbacks import StreamlitCallbackHandler
import streamlit as st

load_dotenv()

st.title("Text To Math Problem Solver")

groq_api = st.sidebar.text_input("Enter Groq API Key", type="password")

if not groq_api:
    st.info("please add api key")
    st.stop()

model = ChatGroq(model="llama-3.1-8b-instant",groq_api_key=groq_api)

wiki = WikipediaAPIWrapper()
wikipedia_tool = Tool(name="wikipedia",func=wiki.run,description="use for general knowledge questions")
math_chain = LLMMathChain.from_llm(llm=model, verbose=True)

def safe_math(question):
    try:
        result = math_chain.invoke({"question": question})
        return result["answer"]
    except Exception as e:
        try:
            return str(eval(question))
        except:
            return "calculation error"

calculator = Tool(name="calculator",func=safe_math,description="use for solving math problems")

prompt = """
You are a math agent.
Solve the problem step by step in simple way.

Question: {question}
Answer:
"""

prompt_template = PromptTemplate(template=prompt,input_variables=["question"])
chain = LLMChain(prompt=prompt_template, llm=model)
reasoning_tool = Tool(name="reasoning",func=chain.run,description="use for word problems and logic")

assistant_agent = initialize_agent(tools=[wikipedia_tool, calculator, reasoning_tool],llm=model,verbose=True,handle_parsing_errors=True)


if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi, I am a Math chatbot"}]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

question = st.text_area("Ask your question")

if st.button("Clear History"):
    st.session_state.messages = []

if st.button("Find My Answer"):
    if question:
        with st.spinner("thinking..."):

            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)

            response = assistant_agent.run(question,callbacks=[st_cb])

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
    else:
        st.warning("enter question first")