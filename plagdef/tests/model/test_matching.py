from plagdef.model.detection import DocumentMatcher
from plagdef.model.matching import VerbatimMatcher
from plagdef.model.models import Document, Cluster, Seed, MatchType


def test_common_words(preprocessor, config):
    doc1 = Document('doc1', 'path/to/doc1',
                    'Some text in doc1. There must be identical sentences. But case or punctuation like this '
                    '"..," do not matter. This is the end.')
    doc2 = Document('doc2', 'path/to/doc2',
                    'Some text in doc2. There must be identical sentences. But Case or punctuation like this'
                    ' \';:\' do not matter. Some different ending.')
    preprocessor.preprocess('eng', [doc1, doc2])
    cluster = Cluster({Seed(doc1.sents(include_common=True)[0], doc2.sents(include_common=True)[0], 0.8, 0.8),
                       Seed(doc1.sents(include_common=True)[1], doc2.sents(include_common=True)[1], 1, 1),
                       Seed(doc1.sents(include_common=True)[2], doc2.sents(include_common=True)[2], 0.4, 0.4)})
    verbatim_matcher = VerbatimMatcher(65)
    match = verbatim_matcher._common_words(cluster).pop()
    assert len(verbatim_matcher._common_words(cluster)) == 1
    # Punctuation included
    assert {(frag.start_char, frag.end_char) for frag in match.frag_pair} == {(19, 107), (19, 108)}


def test_common_words_ends_without_punctuation(preprocessor, config):
    doc1 = Document('doc1', 'path/to/doc1', 'Some text in doc1. There must be identical sentences')
    doc2 = Document('doc2', 'path/to/doc2', 'Some text in doc2. There must be identical sentences')
    preprocessor.preprocess('eng', [doc1, doc2])
    cluster = Cluster({Seed(doc1.sents(include_common=True)[1], doc2.sents(include_common=True)[1], 1, 1)})
    verbatim_matcher = VerbatimMatcher(25)
    match = verbatim_matcher._common_words(cluster).pop()
    assert len(verbatim_matcher._common_words(cluster)) == 1
    # Punctuation included
    assert {(frag.start_char, frag.end_char) for frag in match.frag_pair} == {(19, 52)}


def test_common_words_with_two_substrings(preprocessor, config):
    doc1 = Document('doc1', 'path/to/doc1',
                    'Some text in doc1. There must be identical sentences. SEPARATOR But case or punctuation '
                    'like this "..," do not matter. This is the end.')
    doc2 = Document('doc2', 'path/to/doc2',
                    'Some text in doc2. There must be identical sentences. But Case or punctuation like this'
                    ' \';:\' do not matter. Some different ending.')
    preprocessor.preprocess('eng', [doc1, doc2])
    cluster = Cluster({Seed(doc1.sents(include_common=True)[0], doc2.sents(include_common=True)[0], 0.8, 0.8),
                       Seed(doc1.sents(include_common=True)[1], doc2.sents(include_common=True)[1], 1, 1),
                       Seed(doc1.sents(include_common=True)[2], doc2.sents(include_common=True)[2], 0.4, 0.4)})
    verbatim_matcher = VerbatimMatcher(25)
    matches = verbatim_matcher._common_words(cluster)
    # 1: There must be identical sentences.
    # 2: But case or punctuation like this ';:' OR "..,"
    # 3:              ""                                 do
    # 4:              ""                                    not
    # 5:              ""                                        matter.
    assert len(verbatim_matcher._common_words(cluster)) == 5
    # Punctuation included
    assert all(f in {(frag.start_char, frag.end_char) for frag_pair in {match.frag_pair for match in matches}
                     for frag in frag_pair} for f in {(19, 53), (54, 87), (54, 95), (54, 99), (54, 107)})


def test_resolve_overlaps(preprocessor, config):
    doc1 = Document('doc1', 'path/to/doc1',
                    'Some text in doc1. There must be identical sentences. SEPARATOR But case or punctuation '
                    'like this "..," do not matter. This is the end.')
    doc2 = Document('doc2', 'path/to/doc2',
                    'Some text in doc2. There must be identical sentences. But Case or punctuation like this'
                    " ';:' do not matter. Some different ending.")
    preprocessor.preprocess('eng', [doc1, doc2])
    cluster = Cluster({Seed(doc1.sents(include_common=True)[0], doc2.sents(include_common=True)[0], 0.8, 0.8),
                       Seed(doc1.sents(include_common=True)[1], doc2.sents(include_common=True)[1], 1, 1),
                       Seed(doc1.sents(include_common=True)[2], doc2.sents(include_common=True)[2], 0.4, 0.4)})
    config['min_verbatim_match_char_len'] = 15
    verbatim_matcher = VerbatimMatcher(15)
    matches = verbatim_matcher._common_words(cluster)
    res_matches = VerbatimMatcher._resolve_match_overlaps(matches)
    # 1: There must be identical sentences
    # 2: But case or punctuation like this ';:' OR "..," do not matter
    assert len(res_matches) == 2
    assert {frag.text for frag_pair in {match.frag_pair for match in res_matches} for frag in frag_pair} \
           == {'There must be identical sentences.', "But Case or punctuation like this ';:' do not matter.",
               'But case or punctuation like this "..," do not matter.'}


def test_verbatim_matches(preprocessor, config):
    doc1 = Document('doc1', 'path/to/doc1', 'Some identical text. This is a sentence. This as well. More similar text.')
    doc2 = Document('doc2', 'path/to/doc2',
                    'Some identical text. Totally different words. These are too. More similar text.')
    preprocessor.preprocess('eng', [doc1, doc2])
    clusters = {Cluster({Seed(doc1.sents(include_common=True)[0], doc2.sents(include_common=True)[0], 1, 1)}),
                Cluster({Seed(doc1.sents(include_common=True)[3], doc2.sents(include_common=True)[3], 1, 1)})}
    verbatim_matcher = VerbatimMatcher(5)
    matches = verbatim_matcher.find_verbatim_matches(clusters)
    assert len(matches) == 2
    assert {frag.text for frag_pair in {match.frag_pair for match in matches} for frag in frag_pair} \
           == {'Some identical text.', "More similar text."}


def test_find_matches_with_verbatim_case(config):
    doc1 = Document('doc1', 'path/to/doc1', 'Some identical text. This is a sentence. This as well. More similar text.')
    doc2 = Document('doc2', 'path/to/doc2',
                    'Some identical text. Totally different words. These are too. More similar text.')
    config['min_verbatim_match_char_len'] = 5
    doc_matcher = DocumentMatcher(config)
    doc_matcher.preprocess('eng', {doc1, doc2})
    doc_pair_matches = doc_matcher.find_matches({doc1, doc2})
    assert len(doc_pair_matches.pop().list(MatchType.VERBATIM)) == 2


def test_find_matches_with_verbatim_case_removes_identical_intelligent_case(config):
    doc1 = Document('doc1', 'path/to/doc1', 'Some absolutely identical text. An awesome sentence.')
    doc2 = Document('doc2', 'path/to/doc2', 'Some absolutely identical text. This is nothing alike.')
    config['min_verbatim_match_char_len'] = 5
    doc_matcher = DocumentMatcher(config)
    doc_matcher.preprocess('eng', {doc1, doc2})
    doc_pair_matches = doc_matcher.find_matches({doc1, doc2})
    assert len(doc_pair_matches) == 1
    assert len(doc_pair_matches.pop().list(MatchType.VERBATIM)) == 1


def test_find_matches_with_intelligent_case(config):
    doc1 = Document('doc1', 'path/to/doc1',
                    'Some identical text. This is a sentence. This as well. And another one. Here is more '
                    'similar text.')
    doc2 = Document('doc2', 'path/to/doc2',
                    'Some almost identical text. Totally different words. These are too. Absolutely not '
                    'equal, not a bit.'
                    ' Here is more fairly similar text.')
    config['min_verbatim_match_char_len'] = 15
    doc_matcher = DocumentMatcher(config)
    doc_matcher.preprocess('eng', {doc1, doc2})
    doc_pair_matches = doc_matcher.find_matches({doc1, doc2})
    assert len(doc_pair_matches.pop().list(MatchType.INTELLIGENT)) == 2


def test_find_matches_with_summary_case(config):
    doc1 = Document('doc1', 'path/to/doc1',
                    "Plagiarism is the representation of another author's language, words, thoughts, ideas, "
                    "or expressions as one's own original work. Plagiarism is considered a severe violation of "
                    "academic integrity and a severe breach of journalistic ethics. Plagiarism might not be the same "
                    "in all countries of the world. Some countries, such as the countries India and Poland, "
                    "consider plagiarism to be a crime, and there have been some cases in countries of people being "
                    "imprisoned for plagiarizing. Although plagiarism in some contexts is considered theft or "
                    "stealing, the concept does not exist in a legal sense, although the use of someone else's work "
                    "in order to gain academic credit may meet some legal definitions of fraud. Within academia, "
                    "plagiarism by the students, professors, or researchers is considered academic dishonesty or "
                    "academic fraud, and offenders are subject to academic censure, up to and including expulsion. "
                    "For the professors and researchers, plagiarism is punished by sanctions ranging from suspension "
                    "to termination, along with the loss of credibility and perceived integrity. Predicated upon an "
                    "expected level of comprehensive learning and comprehensive comprehension having been achieved, "
                    "all associated academic accreditation becomes seriously undermined if plagiarism is allowed to "
                    "become the norm within academic submissions. In the academic world, plagiarism by students is "
                    "usually considered a very serious offense that can result in punishments such as a failing grade "
                    "on the particular assignments, the entire course, or even being expelled from the institution.")
    doc2 = Document('doc2', 'path/to/doc2',
                    "Plagiarism are ideas of another passed of as original work. It breaks ethics. It is not equal. "
                    "India considers plagiarism a crime and people were imprisoned. Plagiarism is sometimes "
                    "theft but in a legal sense fraud.Plagiarism by students is dishonesty and offenders are subject "
                    "to censure. For professors, plagiarism is punished from suspension to termination. Academic "
                    "accreditation is undermined if plagiarism is norm within submissions. Plagiarism by student is "
                    "considered a offense that can result in being expelled.")
    config['min_verbatim_match_char_len'] = 256
    config['adjacent_sents_gap'] = 1
    doc_matcher = DocumentMatcher(config)
    doc_matcher.preprocess('eng', {doc1, doc2})
    doc_pair_matches = doc_matcher.find_matches({doc1, doc2})
    assert len(doc_pair_matches[0].list(MatchType.INTELLIGENT)) == 2
    assert len(doc_pair_matches[0].list(MatchType.SUMMARY)) == 1


def test_find_matches_without_documents_returns_empty_list(config):
    doc_matcher = DocumentMatcher(config)
    matches = doc_matcher.find_matches(set())
    assert len(matches) == 0


def test_find_matches_returns_a_match(config):
    docs = {Document('doc1', 'path/to/doc1', 'This is an awesome document. And some text in it.\n'),
            Document('doc2', 'path/to/doc2', 'It\'s a great one. This is an awesome document.\n')}
    doc_matcher = DocumentMatcher(config)
    doc_matcher.preprocess('eng', docs)
    matches = doc_matcher.find_matches(set(docs))
    assert len(matches) == 1


def test_find_matches_with_archive_docs(config):
    docs = {Document('doc1', 'path/to/doc1', 'This is an awesome document. And some text in it.\n'),
            Document('doc2', 'path/to/doc2', 'It\'s a great one. This is an awesome document.\n')}
    archive_docs = {Document('doc3', 'path/to/doc3',
                             'This is an awesome document. This sentence is equal. But it does not matter.'),
                    Document('doc4', 'path/to/doc4',
                             'Similarity of next sentence is ignored. This sentence is equal. And '
                             'some text in it.'),
                    Document('doc5', 'path/to/doc5',
                             'This sentence is equal. And this is fantastic.')}
    doc_matcher = DocumentMatcher(config)
    doc_matcher.preprocess('eng', docs)
    doc_matcher.preprocess('eng', archive_docs)
    matches = doc_matcher.find_matches(set(docs), archive_docs=archive_docs)
    assert len(matches) == 4
