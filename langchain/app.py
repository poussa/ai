from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

#url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
url = "https://akatemia-sports.fi"
parser = StrOutputParser()

messages = [
    SystemMessage(content="Translate the following from English into Italian"),
    HumanMessage(content="hi!"),
]

model = Ollama(model="llama3.1")

def invoke(message):
    result = model.invoke(message)
    msg = parser.invoke(result)
    print(msg)

def loadURL(url):
    loader = WebBaseLoader(url)
    data = loader.load()
    return data

def embeddings(text):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    result = embeddings.embed_query(text)
    print(result)

if __name__ == "__main__":
    #invoke(messages)
    embeddings("this is a test document")