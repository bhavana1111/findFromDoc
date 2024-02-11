#first we seperate the text file into chunks
#Calculate embeddings for Each chunk
from langchain_community.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

embeddings=OpenAIEmbeddings()


text_splitter=CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0
)

#load file using langchain
loader=TextLoader("facts.txt")
#here we are passing text file to TextLoader class
# we ar getting document as the result
docs=loader.load_and_split(
    text_splitter=text_splitter
)

#there will be a directory created to store the embeddings
db=Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="emb"
)

results=db.similarity_search(
    "What is an interesting fact about the english langauge?"
)

for result in results:
    print("\n")
    print(result.page_content)
#Embedding model examples are SentenceTransformer,OpenAI Embeddings