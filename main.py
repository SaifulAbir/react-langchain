import re
from dotenv import load_dotenv
from langchain.agents import tool
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.tools import render_text_description
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    return len(text)


class CustomReActParser(ReActSingleInputOutputParser):
    """Custom parser to handle Gemini's tendency to output complete sequences"""

    def parse(self, text: str):
        # Check if text contains both Action and Final Answer (Gemini's behavior)
        if "Action:" in text and "Final Answer:" in text:
            # Extract just the Action part and remove Final Answer
            action_match = re.search(r'Action:\s*(.+)', text)
            action_input_match = re.search(r'Action Input:\s*(.+?)(?:\n|$)', text, re.DOTALL)

            if action_match and action_input_match:
                # Reconstruct text with only the action part
                thought_part = text.split("Action:")[0]
                cleaned_text = f"{thought_part}Action: {action_match.group(1).strip()}\nAction Input: {action_input_match.group(1).strip()}"
                return super().parse(cleaned_text)

        # Use original parser for normal cases
        return super().parse(text)


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    print("Hello ReAct LangChain!")
    tools = [get_text_length]

    template = """
    Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools), tool_names=", ".join([t.name for t in tools])
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0
    )

    # Instantiate the custom parser
    custom_parser = CustomReActParser()

    agent = {"input": lambda x:x["input"]} | prompt | llm | custom_parser

    res = agent.invoke({"input": "What is the length of DOG in characters?"})

    print(res)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
