from sklearn.base import BaseEstimator
from sentence import Sentence
from itertools import zip_longest
import numpy as np
import copy


class Postprocessor():
    """Preprocessor, applies predictions to list of sentences."""

    def __init__(self, affix_dict={}):
        self.affix_dict = affix_dict

    def fit(self, sentences):
        """Populates dictionary with known affix cues based on a list of sentences."""
        for s in sentences:
            for token, tag, negation in zip(s.forms, s.tags, s.negations):
                if tag == Sentence.Tag.A:
                    self.affix_dict[token] = {
                        'cue': negation['cue'],
                        'scope': negation['scope']
                    }

    def transform(self, sentences, labels):
        """Transforms list of sentences and labels,
        to new list of sentences with negations that corresponds to labels.

        Args:
            sentences: list of sentences to process.
            labels: list of predicted labels.

        Returns:
            list of modified sentences.
        """
        result = []
        for sentence, tags in zip(sentences, labels):
            sentence = copy.deepcopy(sentence)
            nodes = sentence.data['nodes']
            tags = list(map(Sentence.Tag, np.argmax(tags, axis=-1)))[:len(nodes)]

            negations = sum(tag in (Sentence.Tag.C, Sentence.Tag.A) for tag in tags)
            sentence.data['negations'] = negations

            for tag, node in zip_longest(tags, nodes, fillvalue=Sentence.Tag.T):

                # If there is no negations
                # remove all negation values
                if not negations:
                    node.pop('negation', None)
                    continue

                # If there is negation, but no negation
                # value in the node create negation dict
                if 'negation' not in node:
                    node['negation'] = [{'id': 0}]

                # Remove original info about negation
                # from node if present
                node['negation'][0].pop('cue', None)
                node['negation'][0].pop('scope', None)

                # Add new negation info to the node
                if tag == Sentence.Tag.F:
                    node['negation'][0]['scope'] = node['form']
                elif tag == Sentence.Tag.C:
                    node['negation'][0]['cue'] = node['form']
                elif tag == Sentence.Tag.A:
                    node['negation'][0].update(
                        self.affix_dict.get(node['form'], {
                            'cue': node['form'],
                            'scope': ''
                        })
                    )

            result.append(sentence)

        return result
