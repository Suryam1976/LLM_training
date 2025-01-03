from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.llms import Ollama

from dotenv import load_dotenv
load_dotenv()

llm_picker = "groq" ### groq

llm_groq = ChatGroq(model_name="llama3-70b-8192")
llm = Ollama(model="gemma2:2b") 


def initialize_prompt_template():
    # Create a prompt template with example sentiment classifications

    prompt_template = PromptTemplate(

        template="""

        Classify the sentiment of the following review as 'positive' or 'negative': 


        * 'The movie was amazing, I loved every minute of it!' - positive

        * 'The food was bland and the service was terrible.' - negative



        Review: {review}

        Sentiment: """,

        input_variables=["review"]

    )
    return prompt_template


# User input
def user_input(review):  
    # Generate the full prompt with the user review
    prompt_template = initialize_prompt_template()
    full_prompt = prompt_template.format(review=review)
    print(full_prompt)

    # Pass this full prompt to your LLM to get the sentiment prediction 
    # (using your preferred LLM integration with LangChain)
    if llm_picker == "ollama":
        response = llm(full_prompt) # If you are using Ollama
    else:
        response = llm_groq.invoke(full_prompt).content # If you are using Groq

    return response

if __name__ == "__main__":
    review = "This restaurant has the best pizza in town!"
    print(user_input(review))  # Output should be something like: "Sentiment: positive" 