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
from langchain.prompts import PromptTemplate


# -----------------------
# Globals
# -----------------------
vectorstore, retriever = None, None
llm = None

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Answer the question below using only the context provided. "
        "Give only the direct answer — do not list multiple questions or extra information. "
        "If the answer is not in the context, respond with 'I don't know'.\n\n"
        "Context: {context}\n\nQuestion: {question}\nAnswer:"
    )
)

# -----------------------
# Load LLM once at startup
# -----------------------
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

# -----------------------
# Process URLs → build embeddings + retriever
# -----------------------
def process_urls(urls_text):
    global vectorstore, retriever
    urls = [u.strip() for u in urls_text.split(",") if u.strip()]
    if not urls:
        return "❌ Please provide at least one URL."

    try:
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=612,
            chunk_overlap=50
        )
        docs = text_splitter.split_documents(data)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5", 
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )

        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        return "✅ URLs processed successfully. You can now ask a question."

    except Exception as e:
        return f"❌ Error processing URLs: {e}"

# -----------------------
# Ask a question
# -----------------------

def extract_first_qa(result_text):
    lines = [line.strip() for line in result_text.split("\n") if line.strip()]
    question_line, answer_line = None, None

    for i, line in enumerate(lines):
        if line.lower().startswith("question:") and question_line is None:
            question_line = line[len("Question:"):].strip()
            # Look for the next line that starts with "Answer:"
            if i + 1 < len(lines) and lines[i + 1].lower().startswith("answer:"):
                answer_line = lines[i + 1][len("Answer:"):].strip()
            break  # only first Q&A

    if question_line and answer_line:
        return question_line, answer_line
    else:
        return None, "I don't know"

def ask_question(question):
    if retriever is None or llm is None:
        return [], "❌ Please process URLs first."
    
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    result = chain({"query": question})

    question_text, answer = extract_first_qa(result["result"])
    sources = [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]

    return [(question_text, answer)], ", ".join(set(sources))


# -----------------------
# Gradio Interface
# -----------------------
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

    question_input = gr.Textbox(
        label="Ask a Question",
        placeholder="Type your question here...",
        lines=1,
        show_label=True
    )

    chat_output = gr.Chatbot(label="Conversation")
    sources_output = gr.Textbox(label="Sources", interactive=False)

    # Button logic
    process_btn.click(
        fn=process_urls,
        inputs=url_input,
        outputs=status_output
    )

    # Trigger ask_question when pressing Enter
    question_input.submit(
        fn=ask_question,
        inputs=question_input,
        outputs=[chat_output, sources_output]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()
