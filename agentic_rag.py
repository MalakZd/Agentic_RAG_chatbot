from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.messages import HumanMessage
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.tools import create_retriever_tool, tool

load_dotenv(override=True)

texts = ["I am Malak Zahid, a second-year engineering student specializing in Data & Artificial Intelligence at the École Supérieure de Technologie de Casablanca."
         "I am a passionate software developer and data analyst with expertise in Python, Java, .NET, React, and Angular, combined with strong database management skills."
         "Through my professional experiences at INWI, I have developed hands-on expertise in Salesforce CRM, IT process management, and most recently, full-stack web development for data-driven applications."
         "I led the development of a centralized web platform for analyzing fiber network data, implementing interactive dashboards, secure data validation, and collaborative multi-user access controls."
]

embedding_model = OpenAIEmbeddings()


vectorstore = Chroma.from_texts(
    texts,
    embedding_model,
    collection_name="resume_collection",
)

retriever = vectorstore.as_retriever()

retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="resume_retriever",
    description="Use this tool to retrieve information from the resume.",

)

@tool
def get_employee_info(name:str):
    """
    Get information about a given employee (name, salary, seniority) .

    """

    return {"name":name, "salary":50000, "seniority":"5"}

@tool
def send_email(email:str, subject:str, content:str):
    """
    Send an email to a given email address with a subject and content.

    """
    print(f"Email sent to {email} with subject : {subject} and content : {content}")

    return "Email sent successfully"

llm = ChatOpenAI(model="gpt-4o", temperature=0)
                 
agent = create_agent(
    model=llm,
    tools=[get_employee_info, send_email, retriever_tool],
    system_prompt="answer to the user question using the tools provided. If you need to get information about an employee, use the get_employee_info tool. If you need to send an email, use the send_email tool. Always use the tools when necessary and provide a final answer to the user.",
)

# resp = agent.invoke(input={"messages":[HumanMessage(content="What is the salary of Malak?")]})
# print(resp['messages'][-1].content)








  
