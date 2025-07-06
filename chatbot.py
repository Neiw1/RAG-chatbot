import os
import pandas as pd
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Load documents
df = pd.read_csv("documents.csv")
documents = df["content"].tolist()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
chunks = text_splitter.create_documents(documents)

# Create embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

# Create vector store
vector_store = FAISS.from_documents(chunks, embedding=embeddings)
vector_store.save_local("faiss_index")

# Create prompt template
prompt_template = """
Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
Context:\n {context}?\n
Question: \n{question}\n

Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Create RAG chain
model = GoogleGenerativeAI(model="gemini-pro", google_api_key=api_key, temperature=0.3)
chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Guardrail for inappropriate questions
def is_inappropriate(question):
    inappropriate_keywords = ["politics", "political", "sensitive", "unethical", "harmful"]
    return any(keyword in question.lower() for keyword in inappropriate_keywords)

# Chatbot interaction
def chatbot():
    print("Chatbot is ready. Type 'exit' to quit.")
    while True:
        user_question = input("You: ")
        if user_question.lower() == "exit":
            break
        if is_inappropriate(user_question):
            print("Chatbot: This question is inappropriate. Please ask another question.")
            continue
        
        docs = vector_store.similarity_search(user_question)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        print(f"Chatbot: {response['output_text']}")

if __name__ == "__main__":
    chatbot()