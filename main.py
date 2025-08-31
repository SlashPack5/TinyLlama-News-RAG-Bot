import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
import gradio as gr
import torch
from langchain.chains import RetrievalQA

# Globals
vectorstore, retriever, llm = None, None, None

# -----------------------
def process_urls(urls_text):
    global vectorstore, retriever, llm
    urls = [u.strip() for u in urls_text.split(",") if u.strip()]
    if not urls:
        return "❌ Please provide at least one URL."

    try:
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=612
        )
        docs = text_splitter.split_documents(data)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5", 
            model_kwargs={'device': device},      # or 'cpu' if no GPU
            encode_kwargs={'normalize_embeddings': True}
        )

        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

        repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            device_map="auto",
        )

        text_gen_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=200,
            temperature=0.1
        )
        llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

        return "✅ URLs processed successfully. You can now ask a question."

    except Exception as e:
        return f"❌ Error processing URLs: {e}"


# -----------------------
def ask_question(question):
    if retriever is None or llm is None:
        return [], "❌ Please process URLs first."
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )
    result = chain.run(question)
    return [(question, result)], ""

# -----------------------
# Gradio Interface
# -----------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h2>🦙 TinyLlama RAG Bot</h2>")

    with gr.Row():
        url_input = gr.Textbox(
            label="Enter URLs (comma-separated)",
            placeholder="https://example.com, https://another.com"
        )
        process_btn = gr.Button("Process URLs")

    status_output = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        question_input = gr.Textbox(
            label="Ask a Question",
            placeholder="Type your question here..."
        )
        ask_btn = gr.Button("Ask")

    chat_output = gr.Chatbot(label="Conversation")
    sources_output = gr.Textbox(label="Sources", interactive=False)

    # Button logic
    process_btn.click(
        fn=process_urls,
        inputs=url_input,
        outputs=status_output
    )

    ask_btn.click(
        fn=ask_question,
        inputs=question_input,
        outputs=[chat_output, sources_output]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()

