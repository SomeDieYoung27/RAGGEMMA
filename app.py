import PyPDF2
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.prompts import PromptTemplate
import chainlit as cl

# Global LLM instance
llm = ChatOllama(model="phi3")

@cl.on_chat_start
async def on_chat_start():
    files = None

    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to begin!",
            accept=["application/pdf"],
            max_size_mb=100,
            timeout=180,
        ).send()

    file = files[0]
    
    # Inform the user that processing has started
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    try:
        # Read the PDF file
        pdf = PyPDF2.PdfReader(file.path)
        pdf_text = ""
        for page in pdf.pages:
            pdf_text += page.extract_text()

        print(f"Extracted text length: {len(pdf_text)}")

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        texts = text_splitter.split_text(pdf_text)

        print(f"Number of text chunks: {len(texts)}")

        # Create metadata for each chunk
        metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

        # Create a Chroma vector store
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        docsearch = await cl.make_async(Chroma.from_texts)(
            texts, embeddings, metadatas=metadatas
        )
        
        # Initialize message history for conversation
        message_history = ChatMessageHistory()
        
        # Memory for conversational context
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            chat_memory=message_history,
            return_messages=True,
        )

        # Create a custom prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer:"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # Create a chain that uses the Chroma vector store
        chain = ConversationalRetrievalChain.from_llm(
            llm,
            chain_type="stuff",
            retriever=docsearch.as_retriever(search_kwargs={"k": 4}),  # Retrieve top 4 documents
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )

        # Let the user know that the system is ready
        msg.content = f"Processing `{file.name}` done. You can now ask questions about the PDF!"
        await msg.update()
        
        # Store the chain in user session
        cl.user_session.set("chain", chain)

    except Exception as e:
        error_msg = f"An error occurred while processing the PDF: {str(e)}"
        print(error_msg)
        await cl.Message(content=error_msg).send()


@cl.on_message
async def main(message: cl.Message):
    try:
        # Retrieve the chain from user session
        chain = cl.user_session.get("chain") 
        print(f"Received message: {message.content}")
        
        # Call the chain with user's message content
        res = await chain.ainvoke(message.content)
        print(f"Chain response: {res}")
        
        answer = res["answer"]
        source_documents = res["source_documents"] 
        print(f"Answer: {answer}")
        print(f"Number of source documents: {len(source_documents)}")

        text_elements = []
        
        # Process source documents if available
        if source_documents:
            for source_idx, source_doc in enumerate(source_documents):
                source_name = f"source_{source_idx}"
                text_elements.append(
                    cl.Text(content=source_doc.page_content, name=source_name)
                )
            source_names = [text_el.name for text_el in text_elements]
            
            # Add source references to the answer
            if source_names:
                answer += f"\nSources: {', '.join(source_names)}"
            else:
                answer += "\nNo sources found"
        
        # Generate summary of all source documents
        all_texts = "\n\n".join([doc.page_content for doc in source_documents])
        summary_prompt = f"""Provide a comprehensive summary of the following text, covering all key points and details:

        {all_texts}

        Your summary should be thorough and include all important information from the text."""
        summary_res = await llm.ainvoke(summary_prompt, max_tokens=500)
        summary = summary_res.content

        # Generate key points from all source documents
        key_points_prompt = f"""Extract and list all the key points from the following text. Each point should be concise but informative:

        {all_texts}

        Format the output as a numbered list."""
        key_points_res = await llm.ainvoke(key_points_prompt, max_tokens=1000)
        key_points = key_points_res.content

        # Combine original answer, summary, and key points
        full_response = f"{answer}\n\nSummary:\n{summary}\n\nKey Points from Source Documents:\n{key_points}"
        
        # Send the response back to the user
        await cl.Message(content=full_response, elements=text_elements).send()

    except Exception as e:
        error_msg = f"An error occurred while processing your question: {str(e)}"
        print(error_msg)
        if hasattr(e, 'response'):
            print(f"Response content: {e.response.content}")
        await cl.Message(content=error_msg).send()

# Add a new command handler for uploading a new PDF
@cl.on_message
async def handle_message(message: cl.Message):
    if message.content.lower() == "upload new pdf":
        await on_chat_start()
    else:
        await main(message)