# This model attempts to build the LangChain '' into a UI using Chainlit

# Start model by entering "chainlit run main.py -w" in Terminal
from io import DEFAULT_BUFFER_SIZE
import os
import chromadb
from chromadb.utils import embedding_functions
import chainlit as cl
from chainlit import prompt
from langchain.schema import retriever

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

my_secret = "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"
os.environ["OPENAI_API_KEY"] = my_secret

from langchain.memory import ChatMessageHistory, ConversationBufferMemory


# Load vector database that was persisted earlier and check collection count in it
persist_directory = "Datastore/chroma/"
embedding = OpenAIEmbeddings()
vectordb = Chroma(
  persist_directory = persist_directory,
  embedding_function = embedding,
)
#print(vectordb._collection.count()) #check progress

# ask Question:
# question = "wer ebnet den Weg zu einer klimaneutralen Wirtschaft? Beschränke die Antwort auf maximal 385 Zeichen!"
#docsearch = vectordb.similarity_search(question, k=3)
#print(len(docsearch)) #check progress
#print(docsearch) #check similarity_search Results


# Build Prompt
template = """
   Deine Aufgabe ist es, Texte für die politische Partei 'GRÜNE' zu schreiben. Die Texte formulierst du aus Sicht der Partei als Antwort zu den Vorgaben oder Fragen des Nutzers.
  {question}

    Beim erstellen der Texte verlässt du dich insbesondere auf die Informationen die dir hier zur Verfügung gestellt werden. Versuche deine Antwort möglichst abwechlungsreich zu gestalten ohne dich zu wiederholen oder etwas zu erfinden, das nicht in dem zur Verfügung gestellen Material steht.

  {context}
"""

from langchain.prompts.chat import (
    ChatPromptTemplate,
    ChatMessagePromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

prompt = ChatPromptTemplate.from_messages(
  messages = [
    SystemMessagePromptTemplate.from_template(template),
    HumanMessagePromptTemplate.from_template("{question}")
  ]
)

chain_type_kwargs = {"prompt": prompt}


# Start Chainlit UI

@cl.on_chat_start
async def start():
  #await cl.Message(content = "Hello 👋 ").send()

  message_history = ChatMessageHistory()

  memory = ConversationBufferMemory(
    memory_key="chat_history",
    output_key="answer",
    chat_memory=message_history,
    return_messages=True,
  )

  chain = RetrievalQA.from_chain_type(
    llm = ChatOpenAI(),
    retriever = vectordb.as_retriever(
      search_type="mmr",
      search_kwargs={'k': 5, 'fetch_k': 50}
    ),
    chain_type_kwargs = chain_type_kwargs,
    memory = memory
  )


  cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
  chain = cl.user_session.get("chain")
  cb = cl.AsyncLangchainCallbackHandler()

  res = await chain.acall(message, callbacks = [cb])
  answer = res["answer"]

  text_elements = []

  await cl.Message(
    content = answer,
    elements = text_elements
  ).send()




  """
  chain = ConversationalRetrievalChain.from_llm(
    llm = ChatOpenAI(
      model_name = "gpt-3.5-turbo",
    ),
    chain_type = "stuff",
    retriever = docsearch.as_retriever(
      search_type = "similarity_score_threshold",
      search_kwargs = {"k": 5, 'score_threshold': 0.8}
    ),
    memory = memory,
    cond
  )
  """
  """
  chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm = ChatOpenAI(
      model_name = "gpt-3.5-turbo"
    ),
    chain_type = "refine",
    retriever = docsearch.as_retriever(),
    #chain_type_kwargs = chain_type_kwargs,
  )
  """