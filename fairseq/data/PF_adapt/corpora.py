from . import DATASET_REGISTRY
from . import dataset_register, data_abspath
from . import Corpus, sanity_check


@dataset_register('PF-Adapt-en-de', ['train', 'valid', 'test'])
def ted_en_de(split):
    corpora = []
    langs = ['en', 'de']
    for lang in langs:
        sub_path = 'data/{}_{}/{}.{}'.format(langs[0], langs[1], split, lang)
        corpus = Corpus('PF-Adapt-en-de', data_abspath(sub_path), lang)
        corpora.append(corpus)
    return corpora

@dataset_register('PF-Adapt-en-ar', ['train', 'valid', 'test'])
def ted_en_ar(split):
    corpora = []
    langs = ['en', 'ar']
    for lang in langs:
        sub_path = 'data/{}_{}/{}.{}'.format(langs[0], langs[1], split, lang)
        corpus = Corpus('PF-Adapt-en-ar', data_abspath(sub_path), lang)
        corpora.append(corpus)
    return corpora

@dataset_register('PF-Adapt-en-az', ['train', 'valid', 'test'])
def ted_en_az(split):
    corpora = []
    langs = ['en', 'az']
    for lang in langs:
        sub_path = 'data/{}_{}/{}.{}'.format(langs[0], langs[1], split, lang)
        corpus = Corpus('PF-Adapt-en-az', data_abspath(sub_path), lang)
        corpora.append(corpus)
    return corpora

@dataset_register('PF-Adapt-en-be', ['train', 'valid', 'test'])
def ted_en_be(split):
    corpora = []
    langs = ['en', 'be']
    for lang in langs:
        sub_path = 'data/{}_{}/{}.{}'.format(langs[0], langs[1], split, lang)
        corpus = Corpus('PF-Adapt-en-be', data_abspath(sub_path), lang)
        corpora.append(corpus)
    return corpora

@dataset_register('PF-Adapt-en-gl', ['train', 'valid', 'test'])
def ted_en_gl(split):
    corpora = []
    langs = ['en', 'gl']
    for lang in langs:
        sub_path = 'data/{}_{}/{}.{}'.format(langs[0], langs[1], split, lang)
        corpus = Corpus('PF-Adapt-en-gl', data_abspath(sub_path), lang)
        corpora.append(corpus)
    return corpora

@dataset_register('PF-Adapt-en-he', ['train', 'valid', 'test'])
def ted_en_he(split):
    corpora = []
    langs = ['en', 'he']
    for lang in langs:
        sub_path = 'data/{}_{}/{}.{}'.format(langs[0], langs[1], split, lang)
        corpus = Corpus('PF-Adapt-en-he', data_abspath(sub_path), lang)
        corpora.append(corpus)
    return corpora

@dataset_register('PF-Adapt-en-it', ['train', 'valid', 'test'])
def ted_en_it(split):
    corpora = []
    langs = ['en', 'it']
    for lang in langs:
        sub_path = 'data/{}_{}/{}.{}'.format(langs[0], langs[1], split, lang)
        corpus = Corpus('PF-Adapt-en-it', data_abspath(sub_path), lang)
        corpora.append(corpus)
    return corpora

@dataset_register('PF-Adapt-en-sk', ['train', 'valid', 'test'])
def ted_en_sk(split):
    corpora = []
    langs = ['en', 'sk']
    for lang in langs:
        sub_path = 'data/{}_{}/{}.{}'.format(langs[0], langs[1], split, lang)
        corpus = Corpus('PF-Adapt-en-sk', data_abspath(sub_path), lang)
        corpora.append(corpus)
    return corpora

if __name__ == '__main__':
    def merge(*_as):
        _ase = []
        for a in _as:
            _ase.extend(a)
        return _ase

    ls = []
    for key in DATASET_REGISTRY:
        splits, f = DATASET_REGISTRY[key]
        for split in splits:
            ls.append(f(split))

    _all = merge(*ls)
    sanity_check(_all)

