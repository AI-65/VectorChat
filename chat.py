# Import required libraries
import os
import json
import logging
import functools
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional

# Import the LangChain functionality
from langchain.callbacks.manager import AsyncCallbackManagerForChainRun, CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.llms import OpenAIChat
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import BaseLanguageModel

# Set up logging
logging.basicConfig(filename='logfilechat.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Decorator for logging the start of function execution and any errors that occur
def log_errors_and_execution(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        logging.info(f"Running function {function.__name__}")
        try:
            result = function(*args, **kwargs)
            return result
        except Exception as e:
            logging.error(f"An error occurred in function {function.__name__}: {str(e)}")
            raise e

    return wrapper

# Define the PromptTemplate for condensing a follow-up question into a standalone question in German
CONDENSE_PROMPT = PromptTemplate(
    input_variables=["context"],
    template="""Given the following conversation and a follow up question, rephrase the follow up question 
    to be a standalone question in german.

    Chat History:
    []
    Follow Up Input: {context}
    Standalone question:"""
)

# Define the PromptTemplate for answering questions in the context of Sachenrecht in German
QA_PROMPT = PromptTemplate(
    input_variables=["context"],
    template="""You are a helpful AI assistant for law students with the name IUSTUS. Du hilfst Ihnen Fragen zum 
    Sachenrecht zu beantworten. Use the following pieces from a Sachenrecht-Textbook to answer the question.
    If you don't know the answer, just say you don't know. DO NOT try to make up an answer.If the question is not 
    related to the context, politely respond that you are tuned to only answer questions that are related to the context. 
    Always respond in german.

    {context}

    Helpful answer in markdown:"""
)

# Define a custom Chain class which first condenses the question and then generates an answer
class MyCustomChain(Chain):
    condense_prompt: PromptTemplate
    qa_prompt: PromptTemplate
    llm: BaseLanguageModel
    output_key: str = "answer"

    @property
    def input_keys(self) -> List[str]:
        return ["context"]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]
    
    # The _call method takes in the inputs and run_manager, generates a standalone question, and then generates an answer to the question
    @log_errors_and_execution
    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        # Stage 1: Generate standalone question
        standalone_question_prompt = self.condense_prompt.format_prompt(
            context=inputs["context"]
        )
        standalone_question_response = self.llm.generate_prompt(
            [standalone_question_prompt],
            callbacks=run_manager.get_child() if run_manager else None
        )
        standalone_question = standalone_question_response.generations[0][0].text

        # Stage 2: Generate answer based on standalone question
        answer_prompt = self.qa_prompt.format_prompt(
            context=standalone_question
        )
        answer_response = self.llm.generate_prompt(
            [answer_prompt],
            callbacks=run_manager.get_child() if run_manager else None
        )
        answer = answer_response.generations[0][0].text

        return {self.output_key: answer}

# Function to load the query and the textchunks from a JSON file
@log_errors_and_execution
def load_data() -> str:
    with open('data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    document_context = ' '.join(data['document_context'])
    user_input = data['query']
    return document_context + " " + user_input

# Function to use the Chain to answer questions
@log_errors_and_execution
def condense_and_answer():
    # Load documents from the JSON file
    context = load_data()

    llm = OpenAIChat(temperature=0)

    chain = MyCustomChain(
        condense_prompt=CONDENSE_PROMPT,
        qa_prompt=QA_PROMPT,
        llm=llm,
        memory=ConversationBufferMemory()
    )

    # Run the loop until the user decides to stop
    while True:
        result = chain({"context": context})

        print(result['answer'])

        # Ask the user if they want to continue the conversation
        continue_chat = input("Would you like to ask another question? (yes/no): ")
        if continue_chat.lower() != "yes":
            break

        # Get the next question from the user
        new_question = input("Please enter your next question: ")
        context = context + " " + new_question

# Main function that loads the environment variables and starts the question-answering loop
@log_errors_and_execution
def main():
    load_dotenv()
    condense_and_answer()

if __name__ == "__main__":
    main()
