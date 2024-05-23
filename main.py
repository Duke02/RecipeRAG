from pathlib import Path
import typing as tp
import logging

from fastapi import FastAPI
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack import Pipeline

from save_data import _embed_doc

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

chroma_save_loc: Path = Path('.').resolve() / 'data' / 'recipes-dataset'
chroma_doc_store: ChromaDocumentStore = ChromaDocumentStore(collection_name='recipes',
                                                            persist_path=str(chroma_save_loc),
                                                            embedding_function='SentenceTransformerEmbeddingFunction',
                                                            model_name='all-mpnet-base-v2')
text_embedder: SentenceTransformersTextEmbedder = SentenceTransformersTextEmbedder(model='sentence-transformers/all-mpnet-base-v2')
retriever: ChromaEmbeddingRetriever = ChromaEmbeddingRetriever(chroma_doc_store, top_k=5)

template: str = '''
Hey I'm going to give you some recipes. I want you to look at them to answer a question I'll provide you afterwards.

Recipes that you know about:
{% for doc in documents %}
    {{ doc.content }}
{% endfor %}

We have a user that's asking a question about recipe recommendations. Can you give them a recommendation of a recipe based on the recipes above and their question? Make sure to include the ingredients they will need, as well as the directions to make the dish.

Question: {{question}}
Answer: 
'''
prompt_builder: PromptBuilder = PromptBuilder(template=template)
generator: HuggingFaceLocalGenerator = HuggingFaceLocalGenerator(model='bigscience/bloom',  # Might try google/flan-t5-large
                                                                 task='text-generation',
                                                                 generation_kwargs={'max_new_tokens': 1_000})  # 'document-question-answering')

generator.warm_up()
text_embedder.warm_up()

rag_pipeline: Pipeline = Pipeline()
rag_pipeline.add_component('text_embedder', text_embedder)
rag_pipeline.add_component('retriever', retriever)
rag_pipeline.add_component('prompt_builder', prompt_builder)
rag_pipeline.add_component('llm', generator)
rag_pipeline.connect('text_embedder.embedding', 'retriever.query_embedding')
rag_pipeline.connect('retriever', 'prompt_builder.documents')
rag_pipeline.connect('prompt_builder', 'llm')

app = FastAPI()


@app.get('/get_recipe')
async def get_recipe(query: str) -> tp.Any:
    response: dict[str, dict[str, list[str]]] = rag_pipeline.run({'text_embedder': {'text': query}, 'prompt_builder': {'question': query}})
    return response['llm']


# @app.get("/")
# async def root():
#     return {"message": "Hello World"}
#
#
# @app.get("/hello/{name}")
# async def say_hello(name: str):
#     return {"message": f"Hello {name}"}
