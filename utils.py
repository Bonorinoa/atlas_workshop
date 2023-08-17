import logging
import json
import pandas as pd
import os
import nltk
import spacy
import streamlit as st

from langchain.llms import OpenAI, Cohere
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.agents import load_tools, initialize_agent, AgentType, Tool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank, LLMChainExtractor
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.schema import OutputParserException
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

# REMEMBER TO ADD YOUR API KEYS HERE
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]
os.environ["WOLFRAM_ALPHA_APPID"] = st.secrets["WOLFRAM_ALPHA_APPID"]
os.environ["COHERE_API_KEY"] = st.secrets["COHERE_API_KEY"]
# ----------

# TODO: Fix build_chain function
# TODO: Write generic function to build custom langchain tools (i.e., summarise, suggest, search-chat)
# TODO: Write functions to save and load information from memory
# TODO: Implement asynchronous versions of llm/chain builders

# interestingly the es_core_news_sm dictionary in spanish is better at identifying entities than the english one
# python -m spacy download en_core_web_sm <- run in terminal to download the english dictionary (es_core_news_sm for spanish)
#nlp = spacy.load("en_core_web_sm")

@st.cache_data(max_entries=10, ttl=3600, show_spinner=True)
def download_cache_report(report):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return report

def compute_cost(tokens, engine):
    
    model_prices = {"text-davinci-003": 0.02, 
                    "gpt-3.5-turbo": 0.002, 
                    "gpt-4": 0.03,
                    "cohere-free": 0}
    model_price = model_prices[engine]
    
    cost = (tokens / 1000) * model_price

    return cost

def build_llm(max_tokens: int, 
              temperature: int, 
              provider="cohere"):
    '''
    Function to build a LLM model using lanchain library. 
    Default model is text-davinci-003 for OpenAI provider, but you can change it to any other model depending on the provider's models.
    note that for chat models you would set provider = "ChatOpenAI" for example.
    params:
        max_tokens: int, default 260
        temperature: float, default 0.6
        provider: str, default 'openai'
    return:
        llm: Langchain llm object
    '''
    llm = None
    
    if provider == "openai":
        llm = OpenAI(model_name='text-davinci-003', temperature=temperature, max_tokens=max_tokens)
    elif provider == "ChatOpenAI":
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature, max_tokens=max_tokens)
    elif provider == "cohere":
        llm = Cohere(temperature=temperature, max_tokens=max_tokens)
    
    return llm

def build_llm_tools(tools: list,
                    max_tokens=260, 
                    temperature=0.6, 
                    provider="openai"):
    '''
    Function to build agent (llm + tools) using lanchain library.
    params:
        tools: list of tools
        model_name: str, default 'text-davinci-003'
        max_tokens: int, default 260
        temperature: float, default 0.6
        provider: str, default 'openai'
    return:
        agent: Langchain agent object
    '''
    agent = None
    if provider == "openai":
        llm = build_llm(temperature=temperature, max_tokens=max_tokens, provider=provider)

        tools = load_tools(tools, llm=llm)

        agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        
    return agent

def read_perma4(path: str, return_test_answers=False):
    '''
    Function to read perma4 dataset.
    params:
        path: str to json file
        return_test_answers: bool, default False
    return:
        data: pandas dataframe
    '''
    data = pd.read_json(path)
    
    questions = data['questions']
    
    if return_test_answers:
        return data
    else:
        return questions
    
def memory_to_pandas(memory_path: str):
    '''
    Function to convert memory to pandas dataframe.
    params:
        memory_path: path to memory json file
    return:
        df: pandas dataframe
    '''
    with open(memory_path) as f:
        data = json.load(f)    
    
    return data

def build_report(report_generator_profile: dict,
                 perma_results: pd.DataFrame,
                 user_data: list):
    '''
    Function to initialize and run report generator given the AI profile and perma4 results.
    params:
        report_generator_profile: dict
        perma_results: list
    return:
        report: str
    '''
    questions = perma_results['Questions:']
    demo_answers = perma_results['Answers:']
    
    name = report_generator_profile['name']
    agent_type = report_generator_profile['agent_type']
    personality = report_generator_profile['personality']
    knowledge = report_generator_profile['knowledge']
    tools = report_generator_profile['tools']
    keywords = report_generator_profile['keywords']
    description = report_generator_profile['description']
    max_tokens = report_generator_profile['max_tokens']
    temperature = report_generator_profile['temperature']
    
    report_structure = "1. Positive Emotions \n 2. Engagement \n 3. Relationships \n 4. Meaning \n 5. Accomplishment \n 6. Physical Health \n 7. Mindset \n 8. Work Environment \n 9. Economic Security"
 
    sys_prompt_template = '''You are {name}, an expert in [{knowledge}] with {personality} personality. {description}. You can use the following keywords to help you: {keywords} '''
    task_prompt_template = '''Use the following questions {questions} and responses {demo_answers} to provide a well being assessment of the surveyed object with the following properties {user_data} based on the 9 pillars of Perma+4 framework.
    The output must be a structured, insightful and concise report that associates the responses to the questions with the 9 pillars of Perma+4 framework. 
    Here is an example of the desired structure {report_structure}. 
    
    --{user_name}'s REPORT-- 
    
    '''
    prompt_template = sys_prompt_template + task_prompt_template
    
    prompt = PromptTemplate(input_variables=[
                            "name", "knowledge", "description", "keywords", "user_name", "user_data",
                            "questions", "demo_answers", "personality", "report_structure"
                            ],
                            template=prompt_template)
    
    # default of build_llm is text-davinci-003
    engine = build_llm(max_tokens=max_tokens, temperature=temperature)
    
    chain = LLMChain(llm=engine, prompt=prompt)
    report = chain.run({'name': name,
                          'knowledge': knowledge,
                          'description': description,
                          'user_data':user_data,
                          'keywords': keywords,
                          'user_name':user_data[0],
                          'questions': questions,
                          'demo_answers': demo_answers,
                          'personality': personality,
                          'report_structure': report_structure})
    
    # get number of tokens in report
    tokens = len(report.split())
    # cost of report
    report_cost = compute_cost(tokens, 'text-davinci-003')
    
    return report, report_cost
    

def build_pillar_report(report_generator_profile: dict,
                        pillar: str,
                        perma_results: pd.DataFrame,
                        user_data: list):
    '''
    Function to initialize and run report generator given the AI profile and perma4 results for a specific pillar (workshop).
    params:
        report_generator_profile: dict
        perma_results: list
    return:
        report: str
    '''
    questions = perma_results['Question']
    demo_answers = perma_results['Response']
    
    name = report_generator_profile['name']
    agent_type = report_generator_profile['agent_type']
    personality = report_generator_profile['personality']
    knowledge = report_generator_profile['knowledge']
    tools = report_generator_profile['tools']
    keywords = report_generator_profile['keywords']
    description = report_generator_profile['description']
    max_tokens = report_generator_profile['max_tokens']
    temperature = report_generator_profile['temperature']
    
    #report_structure = "1. Positive Emotions \n 2. Engagement \n 3. Relationships \n 4. Meaning \n 5. Accomplishment \n 6. Physical Health \n 7. Mindset \n 8. Work Environment \n 9. Economic Security"
 
    sys_prompt_template = '''You are {name}, an expert in [{knowledge}] with {personality} personality. {description}.'''
    task_prompt_template = '''Use the following questions {questions} and responses {demo_answers} to provide a well being assessment of the surveyed object with the following properties {user_data} based on the the {pillar} pillar of the Perma+4 framework.
    You can use the following keywords to help you: {keywords}.
    The output must be a structured, insightful and concise report that associates the responses to the questions with the specified pillar of Perma+4 framework. The first paragraph must be a description of the user's profile to inform wellbeing coaches.
    
    --{user_name}'s REPORT-- 
    
    '''
    prompt_template = sys_prompt_template + task_prompt_template
    
    prompt = PromptTemplate(input_variables=[
                            "name", "knowledge", "description", "keywords", "user_name", "user_data",
                            "questions", "demo_answers", "personality", "pillar"
                            ],
                            template=prompt_template)
    
    # default of build_llm is text-davinci-003
    engine = build_llm(max_tokens=max_tokens, temperature=temperature)
    
    chain = LLMChain(llm=engine, prompt=prompt)
    report = chain.run({'name': name,
                          'knowledge': knowledge,
                          'description': description,
                          'user_data':user_data,
                          'keywords': keywords,
                          'user_name':user_data[0],
                          'questions': questions,
                          'demo_answers': demo_answers,
                          'personality': personality,
                          'pillar': pillar})
    
    # get number of tokens in report
    tokens = len(report.split())
    # cost of report
    report_cost = compute_cost(tokens, 'text-davinci-003')
    
    return report, report_cost

def generate_goals(recommender_generator_profile: dict,
                   report: str,
                   provider: str):
    '''
    Function to initialize and run recommender generator given the AI profile and user data.
    params:
        recommender_generator_profile: dict
        user_data: list
    return:
        goals: str
    '''
    name = recommender_generator_profile['name']
    agent_type = recommender_generator_profile['agent_type']
    personality = recommender_generator_profile['personality']
    knowledge = recommender_generator_profile['knowledge']
    tools = recommender_generator_profile['tools']
    keywords = recommender_generator_profile['keywords']
    description = recommender_generator_profile['description']
    max_tokens = recommender_generator_profile['max_tokens']
    temperature = recommender_generator_profile['temperature']
    
    sys_prompt_template = '''You are {name}, an expert in [{knowledge}] with {personality} personality. {description}.'''
    task_prompt_template = '''Use the following insights {report} to suggest three or fours goals for the surveyed object that will maximize his net benefit for the effort required to improve along the dimension analyzed in the report. \n
    
    1. Goal A \n
    
    '''
    
    prompt_template = sys_prompt_template + task_prompt_template
    
    prompt = PromptTemplate(input_variables=["name", "knowledge", "description", 
                                             "report", "personality"],
                            template=prompt_template)
    
    engine = build_llm(max_tokens=max_tokens, temperature=temperature, provider=provider)
    
    chain = LLMChain(llm=engine, prompt=prompt)
    
    goals = chain.run({'name': name,
                    'knowledge': knowledge,
                    'description': description,
                    'report': report,
                    "personality": personality})
    
    tokens_used = len(goals.split())
    goals_cost = compute_cost(tokens_used, 'text-davinci-003')
    
    return goals, goals_cost

# TODO: Add curated data source to complement google search results
def suggest_activities(coach_profile: dict,
                       report: str,
                       goals: str,
                       provider: str):
    '''
    Function to initialize and run coach given the AI profile and user data.
    params:
        coach_profile: dict
        user_data: list
    return:
        activities: str
    '''
    name = coach_profile['name']
    agent_type = coach_profile['agent_type']
    personality = coach_profile['personality']
    knowledge = coach_profile['knowledge']
    tools = coach_profile['tools']
    keywords = coach_profile['keywords']
    description = coach_profile['description']
    max_tokens = coach_profile['max_tokens']
    temperature = coach_profile['temperature']
    
    sys_prompt_template = f'''You are {name}, an expert in [{knowledge}] with a {personality} personality. {description}.'''
    task_prompt_template = f'''Given the following insights {report} and goal recommendations {goals}.
    Recommend two or three concrete activities that will help the surveyed object reformat their habits to achieve or move towards the suggested goals. Provide a description of what the activity entails.
    --
    Prioritize the tools as follows: 
    1. google-serper to research custom and relevant activities to recommend. 
    You can use the following keywords to optimize your search: {keywords}.
    --
    
    Write the suggested activities in bullet point format. Each activity must be clearly described and properly structured. 
    
    1. Activity A \n
      
    '''
    
    tools = load_tools(['google-serper'])
    
    prompt_template = sys_prompt_template + task_prompt_template
    
    engine = build_llm(max_tokens=max_tokens, temperature=temperature, provider=provider)
    coach_agent = initialize_agent(tools,
                                   engine,
                                   agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                   verbose=True, max_iterations=7, early_stopping_method="generate"
                                   )
    try:
        activities = coach_agent.run(prompt_template)
        
    except OutputParserException as e:
        activities='1.'+str(e).split('1.')[1]
    
    tokens_used = len(activities.split())
    activities_cost = compute_cost(tokens_used, 'text-davinci-003')
        
    return activities, activities_cost

# ignore this function, is still on development. Not sure if it will be useful or not.
def run_agent_from_profile(agent_profile: dict, 
                           query: str):
    '''
    Function to build agent from memory using lanchain library.
    params:
        agent_profile: dict
        memory: pandas dataframe
    return:
        agent: Langchain agent object
    '''
    agent = None 
    
    name = agent_profile['name']
    agent_type = agent_profile['agent_type']
    personality = agent_profile['personality']
    knowledge = agent_profile['knowledge']
    tools = agent_profile['tools']
    description = agent_profile['description']
    max_tokens = agent_profile['max_tokens']
    temperature = agent_profile['temperature']
    
    engine = build_llm(model_name='text-davinci-003', 
                       max_tokens=max_tokens, temperature=temperature)
    llm_tools = load_tools(tools, llm=engine)
    
    prompt_template = '''You are {name}. {description}. You have a {personality} personality and {knowledge} knowledge.'''
    prompt = PromptTemplate(input_variables=[name, description, query, personality, knowledge],
                            template=prompt_template)
    
    if agent_type == "zeroShot":
        print(f"Building (zeroShot) {name} agent...")
        zeroShot_agent = initialize_agent(tools=llm_tools, llm=engine, 
                                 agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        
        # Build prompt before running to specify custom agent's prompt using its description, personality, and knowledge.
        ## 
        
        zeroShot_chain = LLMChain(llm=engine,
                                  prompt=prompt)
                                  
        agent_response = zeroShot_chain.run(query)
        
        
        #agent = zeroShot_agent
    
    elif agent_type == "selfAskSearch":
        print(f"Building (selfAskSearch) {name} agent...")
        search = GoogleSerperAPIWrapper()
        # intermediate answer tool
        self_tools = [Tool(name="Intermediate Answer",
                        func=search.run,
                        description="useful for when you need to ask with search")]
        
        sealfAsk_agent = initialize_agent(tools=self_tools, llm=engine, 
                                 agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True)
        agent = sealfAsk_agent
    
    return agent_response

## 08/16/2023
def set_smart_goal(user_data, goal):
    '''
    Function to set smart goal conversationally with LLM.
    params:
        user_data: dict
        goal: str
    return:
        smart_goal: str
    '''
    chatgpt = build_llm(model_name='ChatOpenAI', max_tokens=250, temperature=0.8)
    
    # langchain chat template
    messages = [
            SystemMessage(content="You are a helpful assistant that translates English to French."),
            HumanMessage(content="I love programming.")
                ]
    chatgpt(messages)



def co_build_pillar_report(report_generator_profile: dict,
                            pillar: str,
                            perma_results: pd.DataFrame,
                            user_data: list):
    '''
    Function to initialize and run Cohere report generator given the AI profile and perma4 results for a specific pillar (workshop).
    params:
        report_generator_profile: dict
        perma_results: list
    return:
        report: str
    '''
    questions = perma_results['Question']
    demo_answers = perma_results['Response']
    
    name = report_generator_profile['name']
    agent_type = report_generator_profile['agent_type']
    personality = report_generator_profile['personality']
    knowledge = report_generator_profile['knowledge']
    tools = report_generator_profile['tools']
    keywords = report_generator_profile['keywords']
    description = report_generator_profile['description']
    max_tokens = report_generator_profile['max_tokens']
    temperature = report_generator_profile['temperature']
    
    #report_structure = "1. Positive Emotions \n 2. Engagement \n 3. Relationships \n 4. Meaning \n 5. Accomplishment \n 6. Physical Health \n 7. Mindset \n 8. Work Environment \n 9. Economic Security"
 
    sys_prompt_template = '''You are {name}, an expert in [{knowledge}] with {personality} personality. {description}.'''
    task_prompt_template = '''Use the following questions {questions} and responses {demo_answers} to provide a well being assessment of the surveyed object with the following properties {user_data} based on the the {pillar} pillar of the Perma+4 framework.
    You can use the following keywords to help you: {keywords}. 
    The output must be a structured, insightful and concise report in paragraph format that associates the responses to the questions with the specified pillar of Perma+4 framework. 
    The first paragraph must be a description of the user's profile to inform wellbeing coaches. The second paragraph the analysis of how the user stands in the selected pillar. Proceed step by step.
    
    --{user_name}'s REPORT-- 
    
    '''
    prompt_template = sys_prompt_template + task_prompt_template
    
    prompt = PromptTemplate(input_variables=[
                            "name", "knowledge", "description", "keywords", "user_name", "user_data",
                            "questions", "demo_answers", "personality", "pillar"
                            ],
                            template=prompt_template)
    
    # default of build_llm is text-davinci-003
    engine = build_llm(max_tokens=max_tokens, temperature=temperature, provider='cohere')
    
    chain = LLMChain(llm=engine, prompt=prompt)
    report = chain.run({'name': name,
                          'knowledge': knowledge,
                          'description': description,
                          'user_data':user_data,
                          'keywords': keywords,
                          'user_name':user_data[0],
                          'questions': questions,
                          'demo_answers': demo_answers,
                          'personality': personality,
                          'pillar': pillar})
    
    # get number of tokens in report
    tokens = len(report.split())
    # cost of report
    report_cost = compute_cost(tokens, 'cohere-free')
    
    return report, report_cost

def set_smart_goal(smart_profile: dict,
                   report: str,
                   human_input: str):
    '''
    Function to set smart goal conversationally with LLM.
    params:
        smart_profile: dict
        provider: str
    return:
        smart_goal: str
    '''
    name = smart_profile['name']
    persona = smart_profile['system_prompt']
    temperature = smart_profile['temperature']
    max_tokens = smart_profile['max_tokens']
    
    task_prompt = f"\nYou have the following information about the client you are currently working with {report}.\n"

    # langchain chat template
    messages = [SystemMessage(content=persona + task_prompt),
                HumanMessage(content=human_input + " Prompt me after each component of the SMART goal to continue our conversation.")]
    
    # build chat llm
    chatgpt = build_llm(provider='ChatOpenAI', 
                        max_tokens=max_tokens, temperature=temperature)
    
    # run chat llm
    llm_output = chatgpt(messages).content
    
    # cost of report
    llm_cost = compute_cost(len(llm_output.split()), 'gpt-3.5-turbo')
    
    return llm_output, llm_cost
    