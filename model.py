import os
import traceback
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

# Load environment variables
load_dotenv()

# Configuration
DB_FAISS_PATH = os.getenv("DB_FAISS_PATH", "vectorstore/db_faiss")

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """Set up a custom prompt template for the bot.
    
    Returns:
        PromptTemplate: Custom prompt template.
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    """Set up a retrieval-based question-answering chain.
    
    Args:
        llm: Large Language Model.
        prompt: Prompt template.
        db: Vector database object.
        
    Returns:
        RetrievalQA: Configured question-answering chain.
    """
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt}
                                           )
    return qa_chain

def load_llm():
    """Load a large language model.
    
    Returns:
        CTransformers: Loaded language model.
    """
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

def qa_bot():
    """Initialize the QA bot components.
    
    Returns:
        RetrievalQA: Configured QA chain.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cuda'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

def final_result(query):
    """Get bot's response for the given query.
    
    Args:
        query (str): User's query.
        
    Returns:
        dict: Bot's response.
    """
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

def main():
    """Main function to set up chatbot event handlers and manage interactions."""
    try:
        @cl.on_chat_start
        async def start():
            try:
                chain = qa_bot()
                msg = cl.Message(content="Starting the bot...")
                await msg.send()
                msg.content = "Hi, Welcome to WebTekBot. What is your question?"
                await msg.update()
                cl.user_session.set("chain", chain)
            except Exception as e:
                error_message = f"Error in start function: {str(e)}\n{traceback.format_exc()}"
                print(error_message)
                await cl.Message(content=f"An error occurred: {str(e)}").send()

        @cl.on_message
        async def main(message):
            try:
                chain = cl.user_session.get("chain")
                cb = cl.AsyncLangchainCallbackHandler(
                    stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
                )
                cb.answer_reached = True
                res = await chain.acall(message, callbacks=[cb])
                answer = res["result"]
                sources = res["source_documents"]

                if sources:
                    answer += f"\nSources:" + str(sources)
                else:
                    answer += "\nNo sources found"

                await cl.Message(content=answer).send()
            except Exception as e:
                error_message = f"Error in main function: {str(e)}\n{traceback.format_exc()}"
                print(error_message)
                await cl.Message(content=f"An error occurred: {str(e)}").send()
                
    except Exception as e:
        print(f"Error in main script: {str(e)}\n{traceback.format_exc()}")

# Call the main function to execute the script
main()
