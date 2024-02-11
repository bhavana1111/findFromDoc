#first we seperate the text file into chunks
#Calculate embeddings for Each chunk
from langchain_community.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
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
for doc in docs:
    print(doc.page_content)

#Embedding model examples are SentenceTransformer,OpenAI Embeddings