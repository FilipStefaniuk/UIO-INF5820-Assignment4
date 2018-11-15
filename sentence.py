from enum import IntEnum
import json


class Sentence(object):
    """Object representation of sentence with negation and negation scope."""

    class Tag(IntEnum):
        """Tags used for negation scopes."""
        T = 0
        F = 1
        C = 2
        A = 3

    def __init__(self, data):
        """Creates Sentence object from epe data."""
        self.data = data

    @property
    def forms(self):
        return [node['form'] for node in self.data['nodes']]

    @property
    def lemmas(self):
        return [node['properties']['lemma'] for node in self.data['nodes']]

    @property
    def pos_tags(self):
        return [node['properties']['xpos'] for node in self.data['nodes']]

    @property
    def negations(self):
        return [node['negation'][0] if 'negation' in node else {} for node in self.data['nodes']]

    @property
    def tags(self):
        def get_tag(negation):
            if 'cue' in negation and 'scope' in negation:
                return Sentence.Tag.A
            if 'cue' in negation:
                return Sentence.Tag.C
            if 'scope' in negation:
                return Sentence.Tag.F
            return Sentence.Tag.T

        return [get_tag(negation) for negation in self.negations]

    def to_tt(self):
        """Converts sentence to tt format."""
        return '\n'.join(['\t'.join([token, tag.name])
                         for token, tag in zip(self.forms, self.tags)])

    def to_json(self):
        """Converts sentence to json format."""
        return json.dumps(self.data)

    def __repr__(self):
        return "Sentence(data={})".format(self.data)

    @staticmethod
    def from_json(data):
        """Create Sentence object from json string."""
        return Sentence(json.loads(data))
