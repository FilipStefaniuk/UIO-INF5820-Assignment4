import argparse
import sys

from estimator import Estimator
from sentence import Sentence


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('model_dir', help='directory where the model is saved')
    parser.add_argument('--sentences', type=argparse.FileType(), default='./data/cdd.epe',
                        help='epe file with data to evaluate on')
    parser.add_argument('--metrics', type=argparse.FileType('w'), default=sys.stdout,
                        help='file where to output computed metrics.')
    parser.add_argument('--negations_only', action='store_true', default=False,
                        help='use only sentences with at least one negation.')

    return parser.parse_args()


def main():
    args = get_args()

    estimator = Estimator.load(args.model_dir)

    sentences = [Sentence.from_json(line) for line in args.sentences]
    sentences = [s for s in sentences if not args.negations_only or s.data['negations']]

    estimator.evaluate(sentences, output_file=args.metrics)


if __name__ == '__main__':
    main()
