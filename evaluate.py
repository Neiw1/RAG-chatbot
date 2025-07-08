import pandas as pd
from chatbot import Chatbot

def simple_keyword_match(expected, actual):
    expected_words = set(expected.lower().split())
    actual_words = set(actual.lower().split())
    return len(expected_words.intersection(actual_words)) > 0

class Evaluator:
    def __init__(self, chatbot):
        self.chatbot = chatbot

    def evaluate(self, questions_path):
        questions_df = pd.read_csv(questions_path)
        correct = 0
        for index, row in questions_df.iterrows():
            question = row['question']
            retrieved = self.chatbot.retrieve_relevant_chunks(question)
            actual_answer = self.chatbot.generate_response(question, retrieved)
            print(f"Question: {question}")
            print(f"Actual Answer: {actual_answer}")

            if "answer" in questions_df.columns:
                expected_answer = row['answer']
                print(f"Expected Answer: {expected_answer}")
                if simple_keyword_match(expected_answer, actual_answer):
                    correct += 1
                    print("Result: Correct")
                else:
                    print("Result: Incorrect")
            else:
                if "sorry, I couldn't find any relevant information" in actual_answer.lower():
                    correct += 1
                    print("Result: Correct")
                else:
                    print("Result: Incorrect")
            print("-"*20)
        
        accuracy = correct / len(questions_df)
        return accuracy

if __name__ == '__main__':
    bot = Chatbot('documents.csv')
    bot.chunk_documents()
    bot.embed_and_store()
    evaluator = Evaluator(bot)
    
    print("Evaluating no_answer_questions.csv")
    accuracy = evaluator.evaluate('no_answer_questions.csv')
    print(f"Accuracy on no_answer_questions.csv: {accuracy * 100:.2f}%")

