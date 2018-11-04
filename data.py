import json


class Sentence(object):

    TAG_CUE = 'C'
    TAG_AFFIX = 'A'
    TAG_FALSE = 'F'
    TAG_TRUE = 'T'

    def __init__(self, dictionary):
        self._dict = dictionary

    @property
    def tokens(self):
        return [node['form'] for node in self._dict['nodes']]

    @property
    def tags(self):
        if not self._dict['negations']:
            return [Sentence.TAG_TRUE] * len(self._dict['nodes'])

        tags = []
        for node in self._dict['nodes']:
            if 'cue' in node['negation'][0] and 'scope' in node['negation'][0]:
                tags.append(Sentence.TAG_AFFIX)
            elif 'cue' in node['negation'][0]:
                tags.append(Sentence.TAG_CUE)
            elif 'scope' in node['negation'][0]:
                tags.append(Sentence.TAG_FALSE)
            else:
                tags.append(Sentence.TAG_TRUE)
        return tags

    @staticmethod
    def load_epe(epe_file):
        return [Sentence(json.loads(line)) for line in epe_file]

    @staticmethod
    def as_tt(sentences):
        return '\n'.join(['\t'.join(x) for sentence in sentences
                         for x in zip(sentence.tokens, sentence.tags)])


if __name__ == '__main__':
    with open('./data/toy.epe') as f:
        data = Sentence.load_epe(f)

    print(Sentence.as_tt(data))
