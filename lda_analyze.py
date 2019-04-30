"""LDA sample to analyze livedoor blog corpus (Japanese)
I refered the following blog to implement this code.
    - LDAでブログ記事のトピックを抽出・分類する
      https://ohke.hateblo.jp/entry/2017/11/14/230000
"""

import glob
import re
import urllib.request

import gensim
from gensim.models.ldamodel import LdaModel
from janome.tokenizer import Tokenizer
from janome.charfilter import UnicodeNormalizeCharFilter, RegexReplaceCharFilter
from janome.tokenfilter import POSKeepFilter, LowerCaseFilter, ExtractAttributeFilter
from janome.analyzer import Analyzer


LIVEDOOR_PATH = './KNBC_v1.0_090925/corpus2/*.tsv'
STOPWORDS_URL = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'


def loadLivedoorCorpus():
    """Load livedoor blog corpus"""
    data = {}
    paths = glob.glob(LIVEDOOR_PATH)

    for path in paths:
        with open(path, 'r', encoding='euc-jp') as f:
            tsv = f.read()

            for i, line in enumerate(tsv.split('\n')):
                if line == '':
                    break

                elms = line.split('\t')
                index = elms[0].split('-')[0]
                if not index in data:
                    data[index] = ''
                    continue

                data[index] += elms[1] # concat blog body

    print('# of blogs: {}'.format(len(data)) )
    print('[Blog sample] text(KN203_Kyoto_1): \n{}'.format(data['KN203_Kyoto_1']))

    return data


def initAnalyzer():
    """Prepares janome tokenizer"""
    char_filters = [UnicodeNormalizeCharFilter(),
                        RegexReplaceCharFilter('\d+', '0')]

    tokenizer = Tokenizer(mmap=True) # mmap=True for NEologd

    token_filters = [POSKeepFilter(['名詞', '形容詞', '副詞', '動詞']),
                        LowerCaseFilter(),
                        ExtractAttributeFilter('base_form')]

    return Analyzer(char_filters, tokenizer, token_filters)


def loadStopwords():
    """Loads stopwords list"""
    stopwords = []

    with urllib.request.urlopen(STOPWORDS_URL) as response:
        stopwords = [w for w in response.read().decode().split('\r\n') if w != '']

    return stopwords


def main():
    data = loadLivedoorCorpus()
    analyzer = initAnalyzer()
    stopwords = loadStopwords()
    token_data = {bid: list(analyzer.analyze(body)) for bid, body in data.items()}
    print('# of tokens:', sum([len(tokens) for _, tokens in token_data.items()]))

    dictionary = gensim.corpora.Dictionary(token_data.values())
    dictionary.filter_extremes(no_below=3, no_above=0.4)

    # Vectorizes the corpus
    corpus = [dictionary.doc2bow(words) for words in token_data.values()]

    lda = LdaModel(
            corpus=corpus, 
            num_topics=4, 
            id2word=dictionary, 
            random_state=1)

    # Prints frequent words for each topic
    for top in lda.show_topics():
        print('topic[{}]: {}'.format(top[0], top[1]))

    topic_counts = {
        'Kyoto'   : [0, 0, 0, 0],
        'Gourmet' : [0, 0, 0, 0],
        'Keitai'  : [0, 0, 0, 0],
        'Sports'  : [0, 0, 0, 0]
    }

    for k, v in token_data.items():
        category = k.split('_')[1]
        bow = dictionary.doc2bow(v)
        topics = lda.get_document_topics(bow)

        top_topic = sorted(topics, key=lambda topic:topic[1], reverse=True)[0][0]
        topic_counts[category][top_topic] += 1

    print('Topic counts:\n{}'.format(topic_counts))


if __name__ == '__main__':
    main()
