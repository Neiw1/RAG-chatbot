# RAG Chatbot

This project is an implementation of a Retrieval Augmented Generation (RAG) chatbot as described in the provided problem statement. The chatbot is designed to answer questions based on a given set of documents.

## Features

*   **Document Chunking:** The chatbot can process and chunk documents into smaller, manageable pieces for efficient retrieval.
*   **Vector Embeddings:** It uses a sentence transformer model to generate vector embeddings for the text chunks.
*   **Vector Database:** The chatbot stores and indexes the vector embeddings in a simple in-memory database for fast retrieval.
*   **Question Answering:** It uses a pre-trained question-answering model to generate answers based on the retrieved context.
*   **Evaluation:** The project includes an evaluation script to assess the chatbot's performance on single-passage, multi-passage, and no-answer questions.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    ```
2.  **Create a virtual environment:**
    ```bash
    python3 -m venv .venv
    ```
3.  **Activate the virtual environment:**
    ```bash
    source .venv/bin/activate
    ```
4.  **Install the dependencies:**
    ```bash
    pip3 install -r requirements.txt
    ```
5.  **Run the chatbot:**
    ```bash
    python3 chatbot.py "Your question here"
    ```

## Evaluation

To evaluate the chatbot's performance, run the `evaluate.py` script:

```bash
python3 evaluate.py
```

This will run the evaluation on the `single_passage_answer_questions.csv`, `multi_passage_answer_questions.csv`, and `no_answer_questions.csv` datasets.
