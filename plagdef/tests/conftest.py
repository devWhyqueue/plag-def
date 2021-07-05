import stanza
from pytest import fixture

from plagdef.model.models import DocumentPairMatches, Match, Fragment, MatchType
from plagdef.model.pipeline.preprocessing import Preprocessor, Document
from plagdef.model.pipeline.seeding import SeedFinder


@fixture(scope="session", autouse=True)
def download_nlp_models():
    stanza.download('en', processors='tokenize,mwt,pos,lemma', logging_level='WARN')
    stanza.download('de', processors='tokenize,mwt,pos,lemma', logging_level='WARN')


@fixture(scope='session')
def config():
    return {
        'lang': 'eng', 'ser': True,
        'min_cos_sim': 0.3, 'min_dice_sim': 0.33, 'min_cluster_cos_sim': 0.34,
        'adjacent_sents_gap': 4, 'min_adjacent_sents_gap': 0, 'adjacent_sents_gap_summary': 24,
        'min_verbatim_match_char_len': 256, 'min_sent_number': 1, 'min_sent_len': 3, 'min_cluster_char_len': 15,
        'rem_stop_words': False
    }


@fixture(scope='session')
def preprocessor(config):
    return Preprocessor(config['min_sent_len'], config['rem_stop_words'])


@fixture(scope='session')
def nlp_ger():
    return stanza.Pipeline('de', processors='tokenize,mwt,pos,lemma')


@fixture(scope='session')
def seeder(config):
    return SeedFinder(config['min_cos_sim'], config['min_dice_sim'])


@fixture
def preprocessed_docs(preprocessor):
    doc1 = Document('doc1', 'path/to/doc1',
                    'Plagiarism is not the same as copyright infringement. While both terms may '
                    'apply to a particular act, they are different concepts. Copyright infringement '
                    'is a violation of the rights of a copyright holder, when material whose use is '
                    'restricted by copyright is used without consent.')
    doc2 = Document('doc2', 'path/to/doc2',
                    'Plagiarism is also considered a moral offense against anyone who has provided the '
                    'plagiarist with a benefit in exchange for what is specifically supposed to be original '
                    'content. Plagiarism is not the same as copyright infringement. Acts of plagiarism may '
                    'sometimes also form part of a claim for breach of the plagiarist\'s contract.')
    preprocessor.preprocess('eng', [doc1, doc2])
    return doc1, doc2


@fixture
def matches():
    doc1, doc2 = Document('doc1', 'path/to/doc1', 'This is a document.\n'), \
                 Document('doc2', 'path/to/doc2', 'This also is a document.\n')
    doc1_doc2_matches = DocumentPairMatches(doc1, doc2)
    doc1_doc2_matches.add(Match(MatchType.VERBATIM, Fragment(0, 5, doc1), Fragment(0, 5, doc2)))
    doc1_doc2_matches.add(Match(MatchType.INTELLIGENT, Fragment(5, 10, doc1), Fragment(5, 10, doc2)))
    doc3, doc4 = Document('doc3', 'path/to/doc3', 'This is another document.\n'), \
                 Document('doc4', 'path/to/doc4', 'This also is another document.\n')
    doc3_doc4_matches = DocumentPairMatches(doc3, doc4)
    doc3_doc4_matches.add(Match(MatchType.SUMMARY, Fragment(2, 6, doc3), Fragment(2, 8, doc4)))
    return {doc1_doc2_matches, doc3_doc4_matches}
