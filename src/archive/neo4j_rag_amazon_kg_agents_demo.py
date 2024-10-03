#%%
from src.utils.amazon_kg_demo_helper import GraphOperations, EmbeddingModel
from langchain.graphs import Neo4jGraph
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.schema import AgentAction, AgentFinish
from langchain.prompts import StringPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.agents import AgentOutputParser


from typing import List, Dict, Callable, Union
import re
import json
import os

prompt_template = '''You are an intelligent assistant designed to help users find products in a knowledge graph.

You have access to the following tools:
{tools}

You should use the tools to perform specific actions and find the best matching product for the user. Follow these instructions:

- First, use the "Query" tool to look for entities in the user prompt.
- If you find relevant results, stop there and return the final answer.
- If no results are found, use the "Similarity Search" tool with the user's entire prompt to find related items.
- If you still cannot find any results, ask the user for more details or provide a message indicating no results were found.

Respond in the following format:

Question: the input prompt from the user
Thought: what you want to do
Action: the name of the action/tool to take
Action Input: the input to the action
Observation: the result of the action
... (repeating Thought/Action/Observation as needed)
Thought: I now know the final answer
Final Answer: the best answer to the user's question, including product names and IDs if found.

Rules to follow:
- Do not make up product details; only use the provided tools to find answers.
- Use exact product names and IDs from the database in the final response.
- If no answer is found, return: "Sorry, I didn't find any suitable products."

Available Entity Types: {entity_types}

User prompt:
{input}

{agent_scratchpad}
'''


class KnowledgeGraph:
    def __init__(self):
        # Initialize the Neo4j graph connection
        self.graph = Neo4jGraph(
            url=os.getenv("NEO_DB_HOST_AURA"),
            username=os.getenv("NEO_DB_USER"),
            password=os.getenv("NEO_DB_PASSWORD_AURA")
        )
        self.graph_operations = GraphOperations(graph_connection=self.graph)

    def similarity_search(self, prompt: str, threshold: float = 0.5) -> List[Dict[str, Union[str, float]]]:
        """Perform similarity search using embeddings for the given prompt."""
        embedding = EmbeddingModel.create_embedding(prompt)
        query = '''
                WITH $embedding AS inputEmbedding
                MATCH (p:Product)
                WITH p, vector.similarity.cosine(inputEmbedding, p.embedding) AS similarity
                WHERE similarity > $threshold
                RETURN p, similarity
                '''
        result = self.graph.query(query, params={'embedding': embedding, 'threshold': threshold})

        return [
            {
                "id": r['p']['id'],
                "name": r['p']['name'],
                "similarity": r['similarity']
            } for r in result
        ]

    def query_db(self, params: Dict, query_generator=None, similarity_threshold: float = 0.5) -> List[Dict[str, str]]:
        """Query the database with specific parameters."""
        # If query_generator is not provided, use a default or raise an appropriate exception
        if query_generator is None:
            raise ValueError("query_generator must be provided for generating Cypher queries.")

        # Generate the query using the query_generator
        query = query_generator.create_cypher_query(params, similarity_threshold)

        result = self.graph_operations.query_graph(params, query_generator, similarity_threshold)
        matches = [
            {
                "id": r['p']['id'],
                "name": r['p']['name']
            } for r in result
        ]
        return matches



class CustomPromptTemplate(StringPromptTemplate):
    """Custom prompt template for the agent with dynamic information."""

    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        intermediate_steps = kwargs.pop("intermediate_steps", [])
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += f"{action.log}\nObservation: {observation}\nThought: "
        
        kwargs["agent_scratchpad"] = thoughts

        # Create a tools variable from the list of tools provided if not present
        kwargs["tools"] = kwargs.get("tools", "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools]))

        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = kwargs.get("tool_names", ", ".join([tool.name for tool in self.tools]))
        kwargs["entity_types"] = json.dumps(entity_types, indent=2)

        return self.template.format(**kwargs)

class CustomOutputParser(AgentOutputParser):
    """Custom output parser to handle LLM agent actions."""

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        llm_output = llm_output.strip()
        
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )

        # Parse out the action and action input using regex
        regex = r"Action: (.*?)\nAction Input: (.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        
        action = match.group(1).strip()
        action_input = match.group(2).strip().strip('"')
        
        return AgentAction(tool=action, tool_input=action_input, log=llm_output)

class AgentInteraction:
    """Class that handles interaction with LLM agent."""

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        self.tools = [
            Tool(
                name="Query",
                func=self.knowledge_graph.query_db,
                description="Use this tool to find entities in the user prompt that can be used to generate queries"
            ),
            Tool(
                name="Similarity Search",
                func=self.knowledge_graph.similarity_search,
                description="Use this tool to perform a similarity search with the products in the database"
            )
        ]

        self.prompt = CustomPromptTemplate(
            template=prompt_template,
            tools=self.tools,
            input_variables=["input", "intermediate_steps"],
        )

        self.output_parser = CustomOutputParser()
        self.llm = ChatOpenAI(temperature=0, model="gpt-4")
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)

        self.agent = LLMSingleActionAgent(
            llm_chain=self.llm_chain,
            output_parser=self.output_parser,
            stop=["Observation:"],
            allowed_tools=[tool.name for tool in self.tools]
        )

        self.agent_executor = AgentExecutor.from_agent_and_tools(agent=self.agent, tools=self.tools, verbose=True)

    def interact(self, user_prompt: str):
        try:
            result = self.agent_executor.run(user_prompt)
            print("Agent Interaction Result:", result)
            return result
        except Exception as e:
            print("Error during agent interaction:", str(e))
            return None




def main():
    # Initialize knowledge graph handler
    knowledge_graph = KnowledgeGraph()

    # Demonstrate similarity search
    prompt = "I'm looking for nice curtains"
    print("Similarity Search Results:")
    print(knowledge_graph.similarity_search(prompt))

    # Demonstrate agent interaction
    agent_interaction = AgentInteraction(knowledge_graph)
    user_prompt = "I'm searching for pink shirts"
    agent_interaction.interact(user_prompt)


if __name__ == "__main__":
    main()

# %%
