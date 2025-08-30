# 🦙 TinyLlama RAG Bot

A simple Retrieval-Augmented Generation (RAG) chatbot I built using TinyLlama and LangChain.  
The bot can read content from URLs you provide, store it in a vector database, and answer your questions based on that content. I made it mainly for learning and experimenting with RAG and small LLMs.

---

## Features

- Process multiple URLs at once
- Ask questions and get answers from the content of the URLs
- Gradio interface with a soft theme, easy to use
- Works on GPU if available, otherwise falls back to CPU
- Uses FAISS for vector storage and BAAI/bge-small-en-v1.5 embeddings
- Uses TinyLlama-1.1B-Chat as the LLM for answering questions

---

## How It Works

Process URLs: Enter one or more URLs separated by commas. The bot loads the web pages and splits them into chunks for embedding.

Embeddings & Vector Store: Each chunk is converted into embeddings using HuggingFace BGE embeddings. FAISS stores these vectors so the bot can quickly search for relevant information.

Ask Questions: When you ask a question, the bot retrieves the most relevant chunks from FAISS and uses TinyLlama to generate an answer.

Display Answer: The Gradio interface shows the answer and optionally any sources retrieved.

---

## How to Run

Clone the repo:
git clone https://github.com/your-username/TinyLlama-RAG-Bot.git
cd TinyLlama-RAG-Bot

Install dependencies:
pip install -r requirements.txt

Run the app:
python main.py

Open the Gradio interface in your browser:
Enter URLs (comma-separated)
Click Process URLs
Ask your question in the textbox and click Ask

---

## Notes

Works best with English web pages.  
Automatically uses GPU if available for faster processing.  
TinyLlama is a smaller model, so it might not summarize very long pages perfectly.  
You can adjust k in the retriever or chunk size to improve answers.

---

## About Me

I built this project to experiment with LLMs, RAG, and vector databases.  
It's simple, easy to run, and meant for learning and personal use.

Enjoy! 😎
