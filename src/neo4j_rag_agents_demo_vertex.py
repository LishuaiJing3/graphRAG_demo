#%%

import os
from dotenv import load_dotenv
load_dotenv()

import vertexai
from vertexai.preview import reasoning_engines


from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


PROJECT_ID = os.getenv("PROJECT_DEV")
REGION = os.getenv("PROJECT_REGION")
STAGING_BUCKET = os.getenv("STORAGE_BUCKET")
vertexai.init(
    project=PROJECT_ID,
    location=REGION,
    staging_bucket=STAGING_BUCKET,
)
# %%
## this is a google demo project, thus credentials are exposed
URI = os.getenv('NEO4J_URI', 'neo4j+s://demo.neo4jlabs.com')
USER = os.getenv('NEO4J_USERNAME','companies')
PASSWORD = os.getenv('NEO4J_PASSWORD','companies')
DATABASE = os.getenv('NEO4J_DATABASE','companies')

class LangchainCode:
    def __init__(self):
        self.model_name = "gemini-1.5-pro-preview-0409" #"gemini-pro"
        self.max_output_tokens = 1024
        self.temperature = 0.1
        self.top_p = 0.8
        self.top_k = 40
        self.project_id = PROJECT_ID
        self.location = REGION
        self.uri = URI
        self.username = USER
        self.password = PASSWORD
        self.database = DATABASE
        self.prompt_input_variables = ["query"]
        self.prompt_template="""
            You are a venture capital assistant that provides useful answers about companies, their boards, financing etc.
            only using the information from a company database already provided in the context.
            Prefer higher rated information in your context and add source links in your answers.
            Context: {context}"""

    def configure_qa_rag_chain(self, llm, embeddings):
        qa_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.prompt_template),
            HumanMessagePromptTemplate.from_template("Question: {question}"
                                                      "\nWhat else can you tell me about it?"),
        ])

        # Vector + Knowledge Graph response
        kg = Neo4jVector.from_existing_index(
            embedding=embeddings,
            url=self.uri, username=self.username, password=self.password,database=self.database,
            search_type="hybrid",
            keyword_index_name="news_fulltext",
            index_name="news_google",
            retrieval_query="""
              WITH node as c,score
              MATCH (c)<-[:HAS_CHUNK]-(article:Article)

              WITH article, collect(distinct c.text) as texts, avg(score) as score
              RETURN article {.title, .sentiment, .siteName, .summary,
                    organizations: [ (article)-[:MENTIONS]->(org:Organization) |
                          org { .name, .revenue, .nbrEmployees, .isPublic, .motto, .summary,
                          orgCategories: [ (org)-[:HAS_CATEGORY]->(i) | i.name],
                          people: [ (org)-[rel]->(p:Person) | p { .name, .summary, role: replace(type(rel),"HAS_","") }]}],
                    texts: texts} as text,
              score, {source: article.siteName} as metadata
            """,
        )
        retriever = kg.as_retriever(search_kwargs={"k": 5})

        def format_docs(docs):
          return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {"context": retriever | format_docs , "question": RunnablePassthrough()}
            | qa_prompt
            | llm
            | StrOutputParser()
        )
        return chain

    def set_up(self):
        # Removed redundant imports
        llm = ChatVertexAI(
            model_name=self.model_name,
            max_output_tokens=self.max_output_tokens,
            max_input_tokens=32000,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            project=self.project_id,
            location=self.location,
            response_validation=False,
            verbose=True
        )
        embeddings = VertexAIEmbeddings("textembedding-gecko@001")

        self.qa_chain = self.configure_qa_rag_chain(llm, embeddings)

    def query(self, query):
        return self.qa_chain.invoke(query)

 
from langchain.globals import set_debug
set_debug(False)
    
# %%
# testing locally
lc = LangchainCode()
lc.set_up()

response = lc.query('What are the news about IBM and its acquisitions and who are the people involved?')
print(response)

# %%
