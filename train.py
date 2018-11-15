import sys
import argparse

from emb import load_emb_model
from sentence import Sentence
from estimator import Estimator
from preprocessor import Preprocessor
from postprocessor import Postprocessor
from score import starsem_score


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--training', type=argparse.FileType(), default='./data/cdt.epe',
                        help='epe file with training sentences.')
    parser.add_argument('--validation', type=argparse.FileType(), default='./data/cdd.epe',
                        help='epe file with validation sentences.')
    parser.add_argument('--metrics', type=argparse.FileType("w"), default=sys.stdout,
                        help='where to output metrics computed on validation.')
    parser.add_argument('--mode', choices=['BASELINE', 'BASELINE_C', 'BASELINE_C_POS'], default='BASELINE',
                        help='mode in which model is trained, C adds cue information, POS adds pos tags.')
    parser.add_argument('--arch', choices=['LSTM', 'GRU'], default='LSTM',
                        help='which recurrent network architecture to use.')
    parser.add_argument('--word_vectors', default=None,
                        help='file with word embeddings')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train the model.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate used in adam optimizer.')
    parser.add_argument('--max_len', type=int, default=100,
                        help='maximum length of sentence.')
    parser.add_argument('--hidden_size', type=int, default=200,
                        help='memory size of recurrent neural network.')
    parser.add_argument('--negations_only', action='store_true', default=False,
                        help='if present, only sentences with at least one negation are used.')
    parser.add_argument('--tensorboard', action='store_true', default=False)
    parser.add_argument('--save', default=None, help='directory where to save model.')

    return parser.parse_args()


def main():
    args = get_args()

    # Load sentences.
    training_sentences = [Sentence.from_json(line) for line in args.training]
    validation_sentences = [Sentence.from_json(line) for line in args.validation]

    # If negations_only flag is present,
    # keep only sentences with at least one negation.
    if args.negations_only:
        training_sentences = [s for s in training_sentences if s.data['negations']]
        validation_sentences = [s for s in validation_sentences if s.data['negations']]

    # Initialize pre and post processors.
    preproc = Preprocessor(max_len=args.max_len)
    postproc = Postprocessor()
    preproc.fit(training_sentences + validation_sentences)
    postproc.fit(training_sentences + validation_sentences)

    # Load embedding model.
    emb_model = load_emb_model(args.word_vectors) if args.word_vectors else None

    # Build and train model.
    estimator = Estimator(preproc, postproc, mode=args.mode, arch=args.arch,
                          lr=args.lr, hidden_size=args.hidden_size, emb_model=emb_model)
    estimator.model.summary()
    estimator.fit(training_sentences, validation_sentences, epochs=args.epochs, tensorboard=args.tensorboard)

    if args.save:
        estimator.save(args.save)

    # Evaluate model.
    estimator.evaluate(validation_sentences, output_file=args.metrics)


if __name__ == '__main__':
    main()
