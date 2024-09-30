from langchain.chains import GraphCypherQAChain
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI

def create_cypher_chain(graph):
    graph.refresh_schema()
    return GraphCypherQAChain.from_llm(
        cypher_llm=ChatOpenAI(temperature=0, model_name='gpt-4o-mini'),
        qa_llm=ChatOpenAI(temperature=0),
        graph=graph,
        verbose=True,
        allow_dangerous_requests=True
    )

def initialize_tools(vector_qa, cypher_chain):
    return [
        Tool(
            name="Tasks",
            func=vector_qa.run,
            description="""Useful when you need to answer questions about descriptions of tasks.
            Not useful for counting the number of tasks. Use full question as input."""
        ),
        Tool(
            name="Graph",
            func=cypher_chain.run,
            description="""Useful when you need to answer questions about microservices, their dependencies or assigned people.
            Also useful for any sort of aggregation like counting the number of tasks, etc. Use full question as input."""
        ),
    ]

def initialize_mrkl_agent(tools):
    return initialize_agent(
        tools,
        ChatOpenAI(temperature=0, model_name='gpt-4o-mini'),
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True
    )
