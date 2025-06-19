from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.newspaper4k import Newspaper4k
from phi.agent import Agent, RunResponse
from phi.utils.pprint import pprint_run_response

import os
from dotenv import load_dotenv
load_dotenv()
key=os.environ.get('GROQ_API_KEY')  

'''

agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile",api_key=key),
    # tools=[DuckDuckGo(), Newspaper4k()],
    description="You are a senior NYT researcher writing an article on a topic.",
    instructions=[
        # "For a given topic, search for the top 1 links.",
        # "Then read each URL and extract the article text, if a URL isn't available, ignore it.",
        "Analyse and prepare an NYT worthy article based on the information.",
    ],
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True,
    # debug_mode=True,
)
# response: RunResponse = agent.run("simulation thoery")
# # Print the response in markdown format
pprint_run_response('evolution')

from typing import Iterator

# # Run agent and return the response as a stream
# response_stream: Iterator[RunResponse] = agent.run("Simulation theory", stream=False)
# # Print the response stream in markdown format
# pprint_run_response(response_stream, markdown=True, show_time=True) '''

from phi.agent import Agent

agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile",api_key=key),
    description="You are a famous short story writer asked to write for a magazine",
    instructions=["You are a marvel movie character."],
    markdown=True,
    debug_mode=True,
)
agent.print_response("Tell me a 2 sentence funny story.", stream=False)