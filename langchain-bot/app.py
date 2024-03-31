# Bring in deps
import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.title('Japalura AI chatbot based on Langchain🦜🔗')
# prompt = st.text_input('Hey, do you want to say something to me, my friend')
prompt = st.text_input('The topic you want to draw')

# Prompts template
title_template = PromptTemplate(
    input_variables=['topic'],
    # template='Use your imagination to help me describe {topic} using approximately 30 artistic vocabulary, And your output is a python list'
    template='write me an art piece name for me about {topic}'
)

# Script template
script_template = PromptTemplate(
    # input_variables=['title'],
    input_variables=['title', 'wikipedia_research'],
    template='write me an art criticism based on this title TITLE: {title}, while leveraging this wikipedia research: {wikipedia_research}'
)

# Memory
# memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
title_memory = ConversationBufferMemory(
    input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(
    input_key='title', memory_key='chat_history')

# LLMs
llm = OpenAI(temperature=0.9)
# title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)
title_chain = LLMChain(llm=llm, prompt=title_template,
                       verbose=True, output_key='title', memory=title_memory)
# script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True)
script_chain = LLMChain(llm=llm, prompt=script_template,
                        verbose=True, output_key='script', memory=script_memory)

wiki = WikipediaAPIWrapper()

# sequential_chain = SimpleSequentialChain(chains=[title_chain, script_chain], verbose=True)

# sequential_chain = SequentialChain(
#     chains=[title_chain, script_chain], input_variables=['topic'], output_variables=['title', 'script'], verbose=True)

# title_chain = LLMChain(llm=llm, prompt=title_template)

# Show
# if prompt:
#     response = llm(prompt)
#     st.write(response)

# if prompt:
#     response = title_chain.run(topic=prompt)
#     st.write(response)

# if prompt:
#     response = sequential_chain.run(prompt)
#     st.write(response)

# if prompt:
#     response = sequential_chain({'topic': prompt})
#     st.write(response['title'])
#     st.write(response['script'])

#     with st.expander('Message History'):
#         st.info(memory.buffer)

if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)
    st.write(title)
    st.write(script)

    with st.expander('Title History'):
        st.info(title_memory.buffer)

    with st.expander('Script History'):
        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research'):
        st.info(wiki_research)
