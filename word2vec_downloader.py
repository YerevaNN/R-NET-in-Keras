from __future__ import print_function
import argparse
from os import path

from gensim.scripts.glove2word2vec import glove2word2vec
from keras.utils import get_file


def download(output_path):
    SERVER = 'http://nlp.stanford.edu/data/'
    VERSION = 'glove.840B.300d'

    origin = '{server}{version}.zip'.format(server=SERVER, version=VERSION)
    cache_dir = path.join(path.abspath(path.dirname(__file__)), 'data')

    get_file('/tmp/glove.zip',
             origin=origin,
             cache_dir=cache_dir,
             cache_subdir='',
             extract=True)

    print('Converting Glove to word2vec...', end='')
    glove2word2vec(cache_dir + '/' + VERSION + '.zip', output_path)
    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--word2vec_path',
                        type=str,
                        default='data/word2vec_from_glove_300.vec',
                        help='Word2Vec vectors file path')

    args = parser.parse_args()
    download(args.word2vec_path)
