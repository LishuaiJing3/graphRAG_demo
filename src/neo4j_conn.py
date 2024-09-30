#%%
from neo4j import GraphDatabase

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

host = os.getenv("NEO_DB_HOST")
username = os.getenv("NEO_DB_USER")
password = os.getenv("NEO_DB_PASSWORD")
database = os.getenv("NEO_DATABASE")


class HelloWorldExample:

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def print_greeting(self, message):
        with self.driver.session() as session:
            greeting = session.execute_write(self._create_and_return_greeting, message)
            print(greeting)

    @staticmethod
    def _create_and_return_greeting(tx, message):
        result = tx.run("CREATE (a:Greeting) "
                        "SET a.message = $message "
                        "RETURN a.message + ', from node ' + id(a)", message=message)
        return result.single()[0]

if __name__ == "__main__":
    greeter = HelloWorldExample(host, username, password)
    greeter.print_greeting("hello, world")
    greeter.close()

#
# %%
