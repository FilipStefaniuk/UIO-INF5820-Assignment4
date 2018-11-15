from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
from keras.layers import Input, Concatenate
from keras.models import Model
from sklearn.base import BaseEstimator
from operator import attrgetter
from sentence import Sentence
import numpy as np


class Preprocessor(BaseEstimator):
    """Preprocessor, creates arrays of data that may be used for
    neural classifier from list of sentences.
    """
    def __init__(self, tokenizer=None, pos_tokenizer=None, max_len=50):
        self.max_len = max_len
        self.tokenizer = tokenizer or Tokenizer(filters='', lower=True)
        self.cue_tokenizer = Tokenizer(filters='', lower=False)     # For compatibility with get_emb_layer
        self.pos_tokenizer = pos_tokenizer or Tokenizer(filters='', lower=False)

    def fit(self, sentences):
        """Populate tokenizer dictionaries."""
        self.tokenizer.fit_on_texts(map(attrgetter('forms'), sentences))
        self.cue_tokenizer.fit_on_texts([['cue', 'not_cue']])
        self.pos_tokenizer.fit_on_texts(map(attrgetter('pos_tags'), sentences))

    def transform(self, sentences):
        """Transform list of sentences to data for model."""
        cues = np.zeros(shape=(len(sentences), self.max_len))
        tokens = np.zeros(shape=(len(sentences), self.max_len))
        pos_tags = np.zeros(shape=(len(sentences), self.max_len))
        labels = np.zeros(shape=(len(sentences), self.max_len, len(Sentence.Tag)))

        for i, s in enumerate(sentences):
            for j, (token, pos_tag, tag) in enumerate(zip(s.forms, s.pos_tags, s.tags)):
                if j < self.max_len:
                    if tag in (Sentence.Tag.C, Sentence.Tag.A):
                        cues[i, j] = self.cue_tokenizer.word_index['cue']
                    else:
                        self.cue_tokenizer.word_index['not_cue']
                    tokens[i, j] = self.tokenizer.word_index.get(token.lower(), 1)
                    pos_tags[i, j] = self.pos_tokenizer.word_index.get(pos_tag, 1)
                    labels[i, j, int(tag)] = 1

        return (tokens, cues, pos_tags), labels
