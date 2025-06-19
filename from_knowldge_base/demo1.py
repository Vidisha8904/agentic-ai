from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.model.groq import Groq
from phi.embedder.openai import OpenAIEmbedder
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.lancedb import LanceDb, SearchType
from phi.embedder.huggingface import HuggingfaceCustomEmbedder
from phi.embedder.sentence_transformer import SentenceTransformerEmbedder
import os
from dotenv import load_dotenv
load_dotenv()
key=os.environ.get('GROQ_API_KEY')  

# Create a knowledge base from a PDF
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    # Use LanceDB as the vector database
    vector_db=LanceDb(
        table_name="recipes",
        uri="tmp/lancedb",
        search_type=SearchType.vector,
        embedder=SentenceTransformerEmbedder(),
    ),
)
# Comment out after first run as the knowledge base is loaded
knowledge_base.load()

agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile",api_key=key),
    # Add the knowledge base to the agent
    knowledge=knowledge_base,
    show_tool_calls=True,
    add_history_to_messages=True,
    # Number of historical responses to add to the messages.
    num_history_responses=3,
    markdown=True,
)
agent.print_response("what are the ingredients for chicken and galangal in coconut milk soup", stream=True)
# pprint([m.model_dump(include={"role", "content"}) for m in agent.memory.messages])

# agent.print_response("What was my first message?", stream=True)
# # -*- Print the messages in the memory
# pprint([m.model_dump(include={"role", "content"}) for m in agent.memory.messages])
