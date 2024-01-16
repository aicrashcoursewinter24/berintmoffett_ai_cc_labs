import os
import datetime
from langchain_community.chat_models import ChatAnyscale
from langchain.agents import load_tools
from langchain.agents import initialize_agent, AgentType

from dotenv import load_dotenv

load_dotenv()


def build_simple_agent():
    llm = ChatAnyscale(
        anyscale_api_key=os.getenv("ANYSCALE_API_KEY"),
        model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0.7,
    )
    tools = load_tools(["llm-math", "ddg-search"], llm=llm)
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
    return agent


def check_input(user_input):
    if user_input.lower() in ["exit", "quit", "q"]:
        return False
    return True


def main():
    agent = build_simple_agent()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    lines = []
    while check_input(user_input := input("Enter prompt ('exit' to quit): ")):
        try:
            response = agent.invoke({"input": user_input})
        except Exception as e:
            print("An error occurred with the response", e)
            lines.append(f"User: {user_input}\n")
            lines.append(f"Bot: {e}\n")
            continue

        lines.append(f"User: {user_input}\n")
        bot = response["output"]
        lines.append(f"Bot: {bot}\n")
        print("\t" + bot)

    filename = f"chat_logs/chat_{now}.txt"
    with open(filename, "w") as f:
        f.writelines(lines)

    print("Chat log saved to", filename)


if __name__ == "__main__":
    main()
