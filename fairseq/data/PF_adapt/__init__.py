import os
import warnings
from collections import namedtuple

ENV_VAR= 'ROOT'

ROOT = os.environ.get(ENV_VAR, None)
if ROOT is None:
    warnings.warn(
        "Please define {} in environment variable" .format(ENV_VAR)
    )

DATASET_REGISTRY = {}
def dataset_register(tag, splits):
    def __inner(f):
        DATASET_REGISTRY[tag] = (splits, f)
        return f
    return __inner

def data_abspath(sub_path):
    path = os.path.join(ROOT, sub_path)
    return path

Corpus = namedtuple('Corpus', 'tag path lang')
def sanity_check(collection):
    for corpus in collection:
        pass

from . import corpora

