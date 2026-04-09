from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchResults, ShellTool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.agents import create_agent
from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.utilities.github import GitHubAPIWrapper
from langchain.messages import SystemMessage, HumanMessage

load_dotenv()

chat_model = ChatHuggingFace(llm=HuggingFaceEndpoint(model="deepseek-ai/DeepSeek-V3.2"))
github = GitHubAPIWrapper()
git_toolkit = GitHubToolkit.from_github_api_wrapper(github)

shell_tool = ShellTool()
search_tool = DuckDuckGoSearchResults()
tools = git_toolkit.get_tools()
tools.append(shell_tool)
tools.append(search_tool)

agent = create_agent(chat_model, tools)

events = agent.stream(
    {
        "messages": [
            HumanMessage(content="What is the current status of the project?"),
        ]
    },
    stream_mode="values",
)


for event in events:
    message = event["messages"][-1].pretty_print()
    print(message)
