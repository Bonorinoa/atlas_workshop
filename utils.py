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
              provider: str):
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
        
    elif provider == "ChatGPT3":
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature, max_tokens=max_tokens)
    
    elif provider == "ChatGPT4":
        llm = ChatOpenAI(model_name="gpt-4", temperature=temperature, max_tokens=max_tokens)
    
    elif provider == "cohere":
        llm = Cohere(temperature=temperature, max_tokens=max_tokens)
    
    return llm

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
    engine = build_llm(max_tokens=max_tokens, temperature=temperature, 
                       provider="openai")
    
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
    task_prompt_template = '''Use the following insights {report} to recommend three or fours goals for the surveyed object that will maximize his net benefit for the effort required to improve along the dimension analyzed in the report. 
    The output must be an enumerated list of the goal and why it was recommended. \n
    --
    
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
    Recommend two or three concrete activities that will help the surveyed object reformat their habits to achieve or move towards the suggested goals.
    The output must be an enumerated list of the goal and why it was recommended. \n
    --
      
    '''
    
    search = GoogleSerperAPIWrapper()
    tools = [(Tool(name='Intermediate Answer',
                  func=search.run,
                  description="useful for when you need to ask with search"))]
    
    prompt_template = sys_prompt_template + task_prompt_template
    
    engine = build_llm(max_tokens=max_tokens, temperature=temperature, provider=provider)
    coach_agent = initialize_agent(tools,
                                   engine,
                                   agent=AgentType.SELF_ASK_WITH_SEARCH,
                                   verbose=True, max_iterations=10, early_stopping_method="generate"
                                   )
    try:
        activities = coach_agent.run(prompt_template)
        
    except OutputParserException as e:
        activities='1.'+str(e).split('1.')[1]
    
    tokens_used = len(activities.split())
    activities_cost = compute_cost(tokens_used, 'text-davinci-003')
        
    return activities, activities_cost


## 08/17/2023
def chat_smart_goal(smart_profile: dict,
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
    #name = smart_profile['name']
    persona = smart_profile['system_prompt']
    temperature = smart_profile['temperature']
    max_tokens = smart_profile['max_tokens']
    
    task_prompt = f"\nYou have the following information about the client you are currently working with {report}.\n"

    # langchain chat template
    messages = [SystemMessage(content=persona + task_prompt),
                HumanMessage(content=human_input + " Prompt me after each component of the SMART goal to continue our conversation.")]
    
    # build chat llm
    chatgpt = build_llm(provider='ChatGPT4', 
                        max_tokens=max_tokens, temperature=temperature)
    
    # run chat llm
    llm_output = chatgpt(messages).content
    
    # cost of report
    llm_cost = compute_cost(len(llm_output.split()), 'gpt-4')
    
    return llm_output, llm_cost

def completion_smart_goal(smart_profile: dict,
                          report: str,
                          user_goal: str):
    '''
    One-shot completion of smart goal.
    '''
    smart_goal = ""
    
    persona = smart_profile['system_prompt']
    temperature = smart_profile['temperature']
    max_tokens = smart_profile['max_tokens']
    
    # smart report structure
    output_structure = "1. Specific: \n2. Measurable: \n3. Achievable: \n4. Relevant: \n5. Time-bound: \n"
    
    context = "\nYou have the following information about the client you are currently working with {report}.\n"
    sys_prompt_template = persona + context

    # task prompt to build detailed and formatted SMART goal 
    task_prompt_template = '''The user has the following goal: {user_goal}.\n
    Please write a SMART goal for the user. Break down the user's goal into actionable steps and enumerate each component clearly based on the following structure [{output_structure}]. 
    Proceed step by step. \n
    --
    
    '''   
    # prompt template
    prompt_template = sys_prompt_template + task_prompt_template
    prompt = PromptTemplate(input_variables=["report", "user_goal", "output_structure"],
                            template=prompt_template)
    
    # build llm
    davinci = build_llm(max_tokens=max_tokens, temperature=temperature, 
                        provider='openai')
    
    # build chain
    chain = LLMChain(llm=davinci, prompt=prompt)
    
    smart_goal = chain.run({'report': report,
                            'user_goal': user_goal,
                            'output_structure': output_structure})
    
    # cost of report
    llm_cost = compute_cost(len(smart_goal.split()), 'text-davinci-003')
    
    return smart_goal, llm_cost
    
def completion_obstacles_and_planning(smart_goal: str):
    '''
    Function to identify and plan for potential internal or external obstacles of a given SMART goal. 
    Functionality based on the Obstacles and Planning components of the WOOP framework.
    
    params:
        smart_goal: str
    return:
        obstacles: str
    '''
    
    sys_prompt = "You are a professional wellness coach who has received their certification from the International Coaching Federation."\
                + " You specialize in helping users identify potential obstacles from their SMART goal, and set a plan to address such obstacles, based on the WOOP framework." \
                + " Address the user as if you were their wellbeing coach. \n"
    
    task_prompt = "Write a list of potential internal and external obstacles that may prevent the user from achieving their SMART goal: {smart_goal}. \n -- \n Then, recommend a plan to help the user think about how to overcome each obstacle. Proceed step by step. \n -- \n"
    
    prompt = PromptTemplate(input_variables=["smart_goal"],
                            template=sys_prompt + task_prompt)
    
    davinci = build_llm(max_tokens=350, temperature=0.75, 
                       provider='openai')
    
    chain = LLMChain(llm=davinci, prompt=prompt)
    
    obstacles = chain.run({'smart_goal': smart_goal})
    
    # cost
    cost = compute_cost(len(obstacles.split()), 'text-davinci-003')
    
    return obstacles, cost