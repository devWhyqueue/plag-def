from collections import Counter

from plagdef.model.models import Document
from plagdef.model.pipeline.seeding import Seed, _vectorize_sents


def test_match_returns_nothing_if_not_similar(preprocessor, seeder):
    doc1, doc2 = Document('doc1', 'path/to/doc1', 'This is a document.'), Document('doc2', 'path/to/doc2',
                                                                                   'These are different words however.')
    preprocessor.preprocess('eng', [doc1, doc2])
    match = seeder.seed(doc1, doc2)
    assert not match


def test_match_returns_match_if_similar(preprocessor, seeder):
    # We need a second sentence with some different words because otherwise every term would have zero specificity
    doc1 = Document('doc1', 'path/to/doc1', 'The words are all the same. Even these are.')
    doc2 = Document('doc2', 'path/to/doc2', 'The words are all the same. Even these are.')
    preprocessor.preprocess('eng', [doc1, doc2])
    matches = seeder.seed(doc1, doc2)
    assert Seed(doc1.sents(include_common=True)[0], doc2.sents(include_common=True)[0], 1, 1) in matches


def test_vectorize_sents(preprocessed_docs):
    doc1, doc2 = preprocessed_docs
    _vectorize_sents(doc1, doc2)
    # Lemma error "rights" ignored
    # Example for copyright in third sent:
    # tf-isf = tf x ln(N/sf)
    # tf('copyright') in sent3 = 3
    # N (num of all sents) = 6
    # sf (num of sents containing 'copyright') = 3
    # tf-isf = 3 x ln(6/3) = 2.0794...
    assert [sent.tf_isf_bow for sent in doc1.sents(include_common=True)] == \
           [Counter({'not': 1.0986122886681098, 'same': 1.0986122886681098, 'as': 1.0986122886681098,
                     'copyright': 0.6931471805599453, 'infringement': 0.6931471805599453,
                     'plagiarism': 0.4054651081081644, 'be': 0.1823215567939546, 'the': 0.1823215567939546}),
            Counter({'while': 1.791759469228055, 'both': 1.791759469228055, 'term': 1.791759469228055,
                     'apply': 1.791759469228055, 'particular': 1.791759469228055, 'they': 1.791759469228055,
                     'different': 1.791759469228055, 'concept': 1.791759469228055, 'may': 1.0986122886681098,
                     'to': 1.0986122886681098, 'act': 1.0986122886681098, 'be': 0.1823215567939546}),
            Counter({'use': 3.58351893845611, 'of': 2.1972245773362196, 'copyright': 2.0794415416798357,
                     'violation': 1.791759469228055, 'rights': 1.791759469228055, 'holder': 1.791759469228055,
                     'when': 1.791759469228055, 'material': 1.791759469228055, 'whose': 1.791759469228055,
                     'restrict': 1.791759469228055, 'by': 1.791759469228055, 'without': 1.791759469228055,
                     'consent': 1.791759469228055, 'infringement': 0.6931471805599453,
                     'be': 0.5469646703818638, 'the': 0.1823215567939546})]
