import typing as tp
import os
from pathlib import Path
from datetime import datetime
from itertools import pairwise

from pydantic import BaseModel, ValidationError, computed_field
from chromadb import PersistentClient, Collection
from chromadb.utils import embedding_functions
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer


class RecipeInfo(BaseModel):
    title: str
    date: datetime | None = None
    tags: list[str]
    introduction: str
    ingredients: list[str]
    directions: list[str]

    @computed_field
    @property
    def recipe_id(self) -> str:
        return f'{self.title.replace(" ", "_")}-{hash(self) % (2 ** 10):x}'

    @computed_field
    @property
    def text(self) -> str:
        return f'Title:' + self.title + '\nIngredients:\n- ' + '\n- '.join(self.ingredients) + '\nDirections:\n' + '\n'.join(
            [f'{i + 1}. {d}' for i, d in enumerate(self.directions)])

    def __hash__(self) -> int:
        return hash(self.title) + sum(map(hash, self.directions + self.ingredients + self.tags)) % 1_000_000 + hash(self.introduction)


def clean_text(text: str | list[str]) -> str:
    t: str = ((text if isinstance(text, str) else '\n'.join([line.strip() for line in text]))
              .replace('\n\n', '')
              # .replace('\n', '')
              .replace('\\', '')
              .replace('"', '')
              .strip()
              )
    return t


def parse_recipe(recipe: str) -> RecipeInfo:
    split_recipe: list[str] = recipe.split('---')
    data: str = split_recipe[1]
    desc: str = split_recipe[2]
    while '  ' in desc:
        desc = desc.replace('  ', ' ')
    data_order: list[str] = [heading for heading in data.split() if heading in {'title:', 'date:', 'tags:', 'author:'}]

    def get_section(text: str, first_heading: str, second_heading: str, heading1_loc_override: int | None = None,
                    heading2_loc_override: int | None = None) -> str:
        heading1_loc: int = heading1_loc_override or text.find(first_heading)
        heading2_loc: int = heading2_loc_override or text.find(second_heading)
        section: str = text[heading1_loc + len(first_heading):heading2_loc]
        return clean_text(section)

    data_dict: dict[str, str | list[str]] = {heading1[:-1]: get_section(data, heading1, heading2) for heading1, heading2 in pairwise(data_order)}

    data_dict[data_order[-1]] = get_section(data, data_order[-1], '', heading2_loc_override=len(data))

    data_dict['tags'] = [] if 'tags' not in data_dict else [clean_text(tag) for tag in data_dict['tags'].strip('][').split(',')]

    intro: str = clean_text(get_section(desc, '', '## Ingredients', heading1_loc_override=0))
    ingredients: list[str] = clean_text(get_section(desc, '## Ingredients', '## Directions')).splitlines()
    directions: list[str] = clean_text(get_section(desc, '## Directions', 'Originally published')
                                       if 'Originally published' in desc else
                                       get_section(desc, '## Directions', '', heading2_loc_override=len(desc))).splitlines()
    return RecipeInfo(title=data_dict.get('title'), date=data_dict.get('date'), tags=data_dict['tags'], introduction=intro, ingredients=ingredients,
                      directions=directions)


def _embed_doc(doc: str | list[str]) -> np.ndarray:
    if not hasattr(_embed_doc, 'embedding_model'):
        setattr(_embed_doc, 'embedding_model', SentenceTransformer('all-mpnet-base-v2'))
    return getattr(_embed_doc, 'embedding_model').encode(doc)


if __name__ == '__main__':
    base_directory: Path = Path('.').resolve() / 'data' / 'public-domain-recipes' / 'content'

    chroma_save_loc: Path = Path('.').resolve() / 'data' / 'recipes-dataset'
    chroma_client: PersistentClient = PersistentClient(path=str(chroma_save_loc))
    already_has_collection: bool = 'recipes' in chroma_client.list_collections()
    recipe_collection: Collection = chroma_client.get_or_create_collection('recipes',
                                                                           embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                                                                               'all-mpnet-base-v2'))

    for recipe_path in tqdm(list(base_directory.iterdir()), 'Parsing and saving data...'):
        if recipe_path.stem == '_index.md':
            continue
        with recipe_path.open('r') as f:
            recipe: str = f.read()
        try:
            recipe_info: RecipeInfo = parse_recipe(recipe)
        except ValidationError as e:
            print(f'Got {type(e)} while parsing recipe at path {recipe_path}. ({e})')
            continue
        recipe_collection.add(ids=[recipe_info.recipe_id], documents=[recipe_info.text],
                              metadatas=[{'title': recipe_info.title,
                                          'ingredients': '\n'.join(recipe_info.ingredients),
                                          **{f'tag_{t}': True for t in recipe_info.tags},
                                          'published_date': str(recipe_info.date),
                                          'intro': recipe_info.introduction,
                                          'directions': '\n'.join(recipe_info.directions)}])
    print('Done!')
