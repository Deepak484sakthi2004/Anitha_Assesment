from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_similarity, faithfulness, answer_correctness
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from langchain_community.embeddings import HuggingFaceBgeEmbeddings



model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, google_api_key='')
model_name = "BAAI/bge-base-en"
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

# creating an object for embedding model
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},  # use cuda, if gpu is available
    encode_kwargs=encode_kwargs
)

def evaluator(question,llm_ans,docs,correct_ans):

    data_samples = {
        'question': [question],
        'answer': [llm_ans],
        'contexts' : [[docs]],
        'ground_truth': [correct_ans]
    }

    dataset = Dataset.from_dict(data_samples)

#faithfulness,answer_correctness,
    result = evaluate(
        dataset,
        metrics=[answer_similarity], # machine with less compute power, just try answer_similarity!!, or use open-api-key!!

        llm=model,
        embeddings=embeddings,
    )

    return result['answer_similarity']

