from flask import Flask, request, jsonify, render_template
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import warnings

from Dataparser import main
from scripts.readProcessedData import readParsedData
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from evaluator import evaluator

app = Flask(__name__)

gemini_api = ''
# or  pip install python-dotenv and load_env -> create .env file ....


# Filter out the UserWarning from langchain
warnings.filterwarnings("ignore")

'''
what is RAG: 
retrieval augmented Generative model, means it retrieves the data from the external database 
as a casual language model

important parameters/processes for rag:
    1. collect 'input data' : 
                input data can to of any form, if the data is live streamming data such as X {twitter}feed, 
                youtube data etc.., It has an additional ETL layer added to tranform the data from the various sources.(in case of production usage)

                but for study, small scale like personal,commercial chatbot, pdf reader applications the input data
                is of the form .txt .pdf .docx

    2. parse the input data :
                Data preprocessing is a important part for analysing and learning any ML algorithms, the efficiency and performance
                can be drastically improved by proper data preprocessing

                for the internship purpose, as I have to use a .pdf form of input data, I need to read the input data from the pdf
                since normal pdfReader modules can't handle/ process the input data properly.

                I chose LlamaParser to parse the input data from the .pdf files, link to llama parser
                    'https://cloud.llamaindex.ai/api-key'
                    get the api key and paste it into .env file!!


    3. embedd the input data and store it in a vector database :
                There are several options to embed the data and store it in a vector database

                embedding Modules/tools : langchains OpenAIEmbeddings GoogleEmbeddings, opensource embedding models
                vector databases : dedicated vector databases (chroma, quadratn, marqo) pinecone(commerical) , DB supporting VectorDB(mongoDB, PostgreSQL, redism, elasticsearch)

    4. Use a language model to generate user specific information:
            variaty of LLM available , chose one!!

'''

'''
embedding :
    USING OPENSOURCE EMBEDDING MODEL FROM HUGGING FACE, 
    AND COULD HAVE USED LLMANA2 OR 3 8B LOCALLY DOWNLOADED
    just install the requirements.txt file

    pip install -r requirements.txt

language model: 

    BUT FOR PORTABILITY, I GO WITH GEMINI FREEWARE- MODEL 
    LINK TO ACCESS THE GEMINI API KEY IS : 
        'https://ai.google.dev/gemini-api'

    click on get API key in gemini AI studio
    it redirects to the API dashboard and click on get API key and copy the API key
    paste the api key into the .env file in the same directory!!

vector Database:
    i chose FAISS DB for simple data manipulation

    in order to try mongoDB vector DB
        `https://github.com/Deepak484sakthi2004/RAG_Chatbot_implementation`
        
        this repo is a fullstack web application of a chat and recommendation model using rag 
'''

# ---------------------------------------INPUT DATA -----------------------------------------


# read the available data! from the Resumes Relative Directory
input_data_directory = "data/"
processed_data_directory = "processedData/"
main(input_data_directory)

# efficient extraction of data from pdf!!
parsedData = readParsedData(processed_data_directory)


# -------------------------------------- EMBEDDING -------------------------------------------

# importin the embedding model from hugging face
model_name = "BAAI/bge-base-en"
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

# creating an object for embedding model
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},  # use cuda, if gpu is available
    encode_kwargs=encode_kwargs
)

# convert the text to chuncks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    chunks = text_splitter.split_text(text)
    return chunks

def batch_iterable(iterable, batch_size):
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx:min(ndx + batch_size, l)]

def get_vector_store(text_chunks, batch_size=10):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    # Convert the vector store into a retriever
    retriever = vector_store.as_retriever()
    return retriever

def vectorize():
    if parsedData:
        text_chunks = get_text_chunks(parsedData)
        retirever = get_vector_store(text_chunks)
        print("--------------------------Data vectorized sucessfully----------------------")
        return retirever
    else:
        print("CANNOT VECTORIZE THE DATA")
        return 

# Now `retriever` can be used to search/retrieve documents based on the embeddings    
retirever = vectorize()



# ---------------------------------------- LLM ---------------------------------------------

'''
I have to use an evaluation for the generated output for the llm model
*My* three ways for evaluating the RAG models
    1.) use RAGAS - Ragas is a dedicated framework that helps you evaluate your Retrieval Augmented Generation
         (RAG) pipelines
    2.) Use hallucination and rank_grader, answer_correctness prompt templates to evaluate the RAG.
    3.) Use ML algorithms to evaluate.
'''

# example for using prompt templates for evaluation!!


# -------- hallicunation grader --------
# prompt = PromptTemplate(
#     template = f"""
#                 <|begin_of_text|><|start_header_id|>system<|end_header_id|>
#                 You are a grader assesing whether an answer is grounded in / supported by a set of facts.
#                 Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.\n
#                 Provide the binary score as a JSON with a single key 'score and no premable or explaination.
#                 <!eot_id!><!start_header_id!>user<!end_head_id!>
#                 Here are the facts : \n\n{documents} \n\n
#                 Here is the answer : \n\n{generation} \n\n
#                  <eot_id><start_header_id>assistant<end_head_id>
#                """,
#     input_variables = ["generation","documents"]
# )

# hallucination_grader = prompt | llm || JsonOutputParser()

# hallucination_grader.invoke({
#     "generation": generation,
#     "documents": docs
# })


# ------------------------ LLM EVALUATOR ----------------

# prompt = PromptTemplate(
#     template = f"""
#                 system
#                 You are an evaluator tasked with assessing the quality of an LLM-generated response. Your goal is to evaluate the response based on three criteria: similarity to the expected answer, faithfulness to the source material, and correctness of the information provided.
                
#                 - Similarity: How similar is the LLM-generated response to the expected answer? Consider whether the main points and details match.
#                 - Faithfulness: Does the LLM-generated response accurately reflect the information in the source material? Check for any distortions, fabrications, or omissions.
#                 - Correctness: Is the information in the LLM-generated response factually accurate? Consider whether the facts and data presented are correct.

#                 Provide your evaluation scores as percentages (0 to 100) in JSON format with keys 'similarity', 'faithfulness', and 'correctness'.

#                 <!eot_id!><!start_header_id!>user<!end_head_id!>
#                 Here is the LLM-generated response: \n\n{response} \n\n
#                 Here is the expected answer: \n\n{expected_answer} \n\n
#                 Here is the source material: \n\n{source_material} \n\n
#                 <eot_id><start_header_id>assistant<end_head_id>
#                """,
#     input_variables = ["question","generation", "actual_answer", "documents"]
# )
# evaluator = prompt | llm || JsonOutputParser()

# evaluator.invoke({
#     "question":"user_question",
#     "generation": generation,
#     "actual_answer": actual_answer,
#     "documents": docs
# })


# direct model to answer the questions instead of retrieving the relevant data, refining the outout data with respect to the input prompt!! 
def get_conversational_chain():
    prompt_template = """
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>
                You are an assistant for question-answering tasks.
                Use the following peices of retrieved context to answer the question.If you don't know the answer,just say that you dont't know.
                Use three sentences minimum and keep the answer concise. <eot_id><start_header_id>user<end_head_id>

                Question: {question}
                Context : {context}

                Answer : <eot_id><start_header_id>assistant<end_head_id>
               """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, google_api_key=gemini_api)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# for better retrival , it uses llm to tune the results!!! 
def generate_answer(user_question):
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question,k=2)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"], docs


# def generate_answer(user_question):
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question,k=1)
#     out = "\n\n".join([doc.page_content for doc in docs])    
#     return out, docs

def llm_model(prompt):
    model = genai.GenerativeModel('gemini-1.5-pro')
                                   #temperature=0.3, google_api_key=gemini_api)
    response = model.generate_content("JUST PROVIDE THE PERFECT ANSWER IN THREE POINTS!!\n"+prompt)    
    #print(response)
    return response.text

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, google_api_key=gemini_api)

# while(True):
#     user_question = input('ASK QN TO RAG!!')
#     generated_answer, context = generate_answer(user_question)
#     print("\n\nOUTPUT")
#     print(generated_answer)
#     print("\n\n----------- DOCS-------------\n")
#     combined_docs_content = "\n\n".join([doc.page_content for doc in context])
#     print(combined_docs_content)
#     correct_answer = llm_model(user_question+"\n\n"+ combined_docs_content)
#     print("\n\n----------- GEMINI ANSWER-------------\n")
#     print(correct_answer)

#     evaluate = input("WANNA EVALUATE?")
#     if(evaluate.lower() == 'yes'):
#         print("\n\n-------------EVALUATE-------------\n")
#         print(evaluator(embeddings,model,user_question,generated_answer,combined_docs_content,correct_answer))



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.form['question']
    generated_answer, context = generate_answer(user_question)
    return jsonify({'response': prettify_text(generated_answer), 'doc': prettify_text(context)})

@app.route('/evaluate', methods=['POST'])
def evaluate():
    user_question = request.form['question']
    generated_answer, context = generate_answer(user_question)
    combined_docs_content = "\n\n".join([doc.page_content for doc in context])
    correct_answer = llm_model("Act as a QA bot and respond accordingly\n Question:"+user_question + "\n\n Provided Content:" + combined_docs_content)
    print("\n\nCORRECT ANSWER\n\n",correct_answer)
    evaluation_result = evaluator(user_question, generated_answer, combined_docs_content, correct_answer)
    print("evaluation reult",evaluation_result)
    return jsonify({'evaluation': evaluation_result,'response': prettify_text(generated_answer),'correct_ans': prettify_text(correct_answer),'doc':prettify_text(combined_docs_content)})

def prettify_text(text):
    prettified = text.replace('\n', '<br>')
    prettified = prettified.replace('**', '<b>').replace('*', '<li>')
    prettified = prettified.replace('<b>', '</b>', 1)  # Ensure to close the first bold tag correctly
    return prettified

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')








