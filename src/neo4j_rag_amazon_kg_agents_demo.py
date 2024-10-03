# %%
import os
import json
from typing import Callable
from typing import List, Union
import re

from src.utils.amazon_kg_demo_helper import GraphOperations, EmbeddingModel
from langchain.graphs import Neo4jGraph
from langchain.agents import (
    Tool,
    AgentExecutor,
    LLMSingleActionAgent,
    AgentOutputParser,
)
from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage

from langchain.prompts import StringPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser


graph = Neo4jGraph(
    url=os.getenv("NEO_DB_HOST_AURA"),
    username=os.getenv("NEO_DB_USER"),
    password=os.getenv("NEO_DB_PASSWORD_AURA"),
)
graph_operations = GraphOperations(graph_connection=graph)

entity_types = {
    "product": "Item detailed type, for example 'high waist pants', 'outdoor plant pot', 'chef kitchen knife'",
    "category": "Item category, for example 'home decoration', 'women clothing', 'office supply'",
    "characteristic": "if present, item characteristics, for example 'waterproof', 'adhesive', 'easy to use'",
    "measurement": "if present, dimensions of the item",
    "brand": "if present, brand of the item",
    "color": "if present, color of the item",
    "age_group": "target age group for the product, one of 'babies', 'children', 'teenagers', 'adults'. If suitable for multiple age groups, pick the oldest (latter in the list).",
}


def query_db(params):
    matches = []
    # Querying the db
    result = graph_operations.query_graph(params)
    for r in result:
        product_id = r["p"]["id"]
        matches.append({"id": product_id, "name": r["p"]["name"]})
    return matches


def similarity_search(prompt: str, threshold: float = 0.5) -> list:

    matches = []

    # Generate the embedding for the given prompt
    embedding = EmbeddingModel.create_embedding(prompt)

    # Neo4j query using native vector similarity, including the score in the RETURN statement
    query = """
            WITH $embedding AS inputEmbedding
            MATCH (p:Product)
            WITH p, vector.similarity.cosine(inputEmbedding, p.embedding) AS similarity
            WHERE similarity > $threshold
            RETURN p, similarity
            """

    # Execute the query
    result = graph.query(query, params={"embedding": embedding, "threshold": threshold})

    # Parse the result and collect the matched products with similarity scores
    for r in result:
        product_id = r["p"]["id"]
        matches.append(
            {"id": product_id, "name": r["p"]["name"], "similarity": r["similarity"]}
        )
    return matches


prompt_similarity = "I'm looking for nice curtains"
print(similarity_search(prompt_similarity))

print(
    graph_operations.similarity_search(
        EmbeddingModel.create_embedding(prompt_similarity)
    )
)
# %%
tools = [
    Tool(
        name="Query",
        func=query_db,
        description="Use this tool to find entities in the user prompt that can be used to generate queries",
    ),
    Tool(
        name="Similarity Search",
        func=similarity_search,
        description="Use this tool to perform a similarity search with the products in the database",
    ),
]

tool_names = [f"{tool.name}: {tool.description}" for tool in tools]

prompt_template = """Your goal is to find a product in the database that best matches the user prompt.
You have access to these tools:

{tools}

Use the following format:

Question: the input prompt from the user
Thought: you should always think about what to do
Action: the action to take (refer to the rules below)
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Rules to follow:

1. Start by using the Query tool with the prompt as parameter. If you found results, stop here.
2. If the result is an empty array, use the similarity search tool with the full initial user prompt. If you found results, stop here.
3. If you cannot still cannot find the answer with this, probe the user to provide more context on the type of product they are looking for. 

Keep in mind that we can use entities of the following types to search for products:

{entity_types}.

3. Repeat Step 1 and 2. If you found results, stop here.

4. If you cannot find the final answer, say that you cannot help with the question.

Never return results if you did not find any results in the array returned by the query tool or the similarity search tool.

If you didn't find any result, reply: "Sorry, I didn't find any suitable products."

If you found results from the database, this is your final answer, reply to the user by announcing the number of results and returning results in this format (each new result should be on a new line):

name_of_the_product (id_of_the_product)"

Only use exact names and ids of the products returned as results when providing your final answer.


User prompt:
{input}

{agent_scratchpad}

"""


# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        ############## NEW ######################
        # tools = self.tools_getter(kwargs["input"])
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in tools]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
        kwargs["entity_types"] = json.dumps(entity_types)
        return self.template.format(**kwargs)


prompt = CustomPromptTemplate(
    template=prompt_template,
    tools=tools,
    input_variables=["input", "intermediate_steps"],
)


# %%
class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )

        # Parse out the action and action input
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)

        # If it can't parse the output it raises an error
        # You can add your own logic here to handle errors in a different way i.e. pass to a human, give a canned response
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)

        # Return the action and action input
        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )


output_parser = CustomOutputParser()

llm = ChatOpenAI(temperature=0, model="gpt-4")

# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Using tools, the LLM chain and output_parser to make an agent
tool_names = [tool.name for tool in tools]

agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\Observation:"],
    allowed_tools=tool_names,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)


def agent_interaction(user_prompt):
    agent_executor.run(user_prompt)


prompt1 = "I'm searching for pink shirts"
agent_interaction(prompt1)
# %%
