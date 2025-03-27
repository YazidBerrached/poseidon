import os
from dotenv import main
import pandas as pd
from sqlalchemy import create_engine
import json
from llama_index.core import Document
from llama_index.core.postprocessor import LLMRerank
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import QueryBundle
import pandas as pd
from IPython.display import display, HTML
from llama_index.core.tools import RetrieverTool
from llama_index.core.retrievers import RouterRetriever
from llama_index.core.selectors import PydanticSingleSelector
from mistralai import Mistral
from llama_index.core import Settings
from llama_index.embeddings.mistralai import MistralAIEmbedding

import tiktoken


main.load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
os.environ["MISTRAL_API_KEY"]  = MISTRAL_API_KEY
DATABASE_URL  = os.getenv("db_url")
client = Mistral(api_key=MISTRAL_API_KEY)
engine = create_engine(DATABASE_URL)

def convert_metadata(meta):
    if isinstance(meta, str):  # Convert JSON string to dict
        return json.loads(meta)
    elif isinstance(meta, dict):  # Already a dictionary
        return meta
    else:
        raise ValueError(f"Unexpected metadata format: {meta}")
    
def get_all_text(new_nodes):
    texts = []
    for i, node in enumerate(new_nodes, 1):
        texts.append(f"\n- {node.get_text()}")
    return ' '.join(texts)


def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n", "")))


def visualize_retrieved_nodes(nodes) -> None:
    result_dicts = []
    for node in nodes:
        result_dict = {"Score": node.score, "Text": node.node.get_text()}
        result_dicts.append(result_dict)

    pretty_print(pd.DataFrame(result_dicts))


Settings.embed_model = MistralAIEmbedding(model_name="mistral-embed", api_key=MISTRAL_API_KEY)

df = pd.read_sql("SELECT * FROM ai.pdf_documents", engine)
df=df.drop(columns=['filters', 'usage', 'created_at', 'updated_at', 'content_hash','embedding', 'id','name'])
df["meta_data"] = df["meta_data"].apply(convert_metadata)
docs = [Document(text=row['content'],metadata=row["meta_data"]) for index, row in df.iterrows()]
index = VectorStoreIndex.from_documents(
    docs,
    show_progress = True
)


def get_retrieved_nodes(
    query_str, vector_top_k=10, reranker_top_n=3, with_reranker=False,index=index):
    query_bundle = QueryBundle(query_str)
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=vector_top_k,
    
    )
    retrieved_nodes = retriever.retrieve(query_bundle)

    if with_reranker:
        reranker = LLMRerank(
            choice_batch_size=5,
            top_n=reranker_top_n,
        )
        retrieved_nodes = reranker.postprocess_nodes(
            retrieved_nodes, query_bundle
        )

    return retrieved_nodes



def further_retrieve(query):
    # Retrieve new nodes based on the query
    new_nodes = get_retrieved_nodes(
        query,
        index=index,
        vector_top_k=10,
        reranker_top_n=5,
        with_reranker=False,
    )
   
    
    #visualize_retrieved_nodes(new_nodes)
        
    retriever = index.as_retriever( similarity_top_k=10)
    try : 
        query_bundle = QueryBundle(query)
        retrieved_nodes = retriever.retrieve(query_bundle)
        reranker = LLMRerank(
            choice_batch_size=5, 
            top_n=7 
        )
        
       
        reranked_nodes = reranker.postprocess_nodes(retrieved_nodes, query_bundle)
        return get_all_text(reranked_nodes)
    except :
        print("No rerank")
        return get_all_text(retriever.retrieve(query))


