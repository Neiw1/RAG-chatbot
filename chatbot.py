
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import sys
from transformers import pipeline

class Chatbot:
    def __init__(self, documents_path):
        self.documents = pd.read_csv(documents_path)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_db = {}
        self.chunks = []
        self.qa_pipeline = pipeline('question-answering', model='distilbert-base-cased-distilled-squad', device='cpu')

    def chunk_documents(self):
        for index, row in self.documents.iterrows():
            # Reverted to paragraph-based chunking
            paragraphs = row['text'].split('\n\n')
            for paragraph in paragraphs:
                if paragraph.strip():
                    self.chunks.append({'text': paragraph.strip(), 'source_doc_index': index})
        return self.chunks

    def embed_and_store(self):
        chunk_texts = [chunk['text'] for chunk in self.chunks]
        chunk_embeddings = self.embedder.encode(chunk_texts, convert_to_tensor=True)
        self.vector_db = chunk_embeddings
        return self.vector_db

    def retrieve_relevant_chunks(self, query, top_k=5):
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embedding, self.vector_db)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        return [(self.chunks[i], score) for i, score in zip(top_results[1], top_results[0])]

    def generate_response(self, query, retrieved_chunks):
        context = "\n\n".join([chunk['text'] for chunk, score in retrieved_chunks if score > 0.3])
        if not context:
            return "I'm sorry, I couldn't find any relevant information in the documents to answer your question."

        result = self.qa_pipeline(question=query, context=context)
        return result['answer']

    def guardrail(self, query):
        profanity = ["badword1", "badword2"] # Add more profanity words
        if any(word in query.lower() for word in profanity):
            return "I cannot answer questions containing profanity."
        return None

if __name__ == '__main__':
    bot = Chatbot('documents.csv')
    bot.chunk_documents()
    print(f"Successfully chunked documents into {len(bot.chunks)} chunks.")
    bot.embed_and_store()
    print(f"Successfully embedded and stored {len(bot.chunks)} chunks.")

    if len(sys.argv) > 1:
        user_query = sys.argv[1]
    else:
        user_query = "What is the EU AI Act?"

    guardrail_response = bot.guardrail(user_query)
    if guardrail_response:
        print(f"Chatbot: {guardrail_response}")
    else:
        retrieved = bot.retrieve_relevant_chunks(user_query)
        response = bot.generate_response(user_query, retrieved)
        print(f"Chatbot: {response}")
