from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from keras.layers.embeddings import Embedding
from keras.layers import Input, Concatenate, Dense
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.recurrent import LSTM, GRU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from keras.models import Model, load_model
from enum import IntEnum, Enum
from operator import attrgetter

from emb import get_emb_layer
from score import starsem_score
from sentence import Sentence

import sys
import os
import tempfile
import pickle
import shutil
import numpy as np


class Estimator(object):
    """Used for building, training and evaluating model."""

    class Mode(IntEnum):
        """Mode in which the model is trained."""
        BASELINE = 1
        BASELINE_C = 2
        BASELINE_C_POS = 3

    class Arch(Enum):
        """Type of recurrent network architecture."""
        LSTM = LSTM
        GRU = GRU

    def __init__(self, preprocessor, postprocessor, mode='BASELINE', arch='LSTM', **kwargs):
        """Initialize Estimator object.

        Args:
            preprocessor: used for conversion from sentences to trainig data.
            postprocessor: converts predicted labels back to epe sentences.
            mode: mode in which model is trained.
            arch: recurrent neural network architecture.
            **kwargs: arguments passed to _build_model()
        """
        self.mode = Estimator.Mode[mode]
        self.arch = Estimator.Arch[arch]
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self._build_model(**kwargs)

    @staticmethod
    def tag_score(y_true, y_pred, mask=None, file=sys.stdout):
        """Computes metrics for tagging accuracy."""
        y_true = np.argmax(y_true, axis=-1)
        y_pred = np.argmax(y_pred, axis=-1)

        if mask is not None:
            y_true = y_true[mask]
            y_pred = y_pred[mask]

        y_true = y_true.ravel()
        y_pred = y_pred.ravel()

        avg_prec, avg_rec, avg_f1, avg_sup = precision_recall_fscore_support(y_true, y_pred, average='macro')

        print(classification_report(y_true, y_pred,
              target_names=[Sentence.Tag(x).name for x in range(len(Sentence.Tag))]),
              file=file)
        print('avg / macro       {}      {}      {}      {}\n'.format(
              np.round(avg_prec, 2), np.round(avg_rec, 2), np.round(avg_f1, 2), avg_sup),
              file=file)
        print('Accuracy: {}'.format(np.round(accuracy_score(y_true, y_pred), 3)),
              file=file)

    def _build_model(self, lr=1e-4, hidden_size=200, emb_model=None):
        """Builds model according to estimator mode and arch.

        Args:
            lr: learning rate.
            hidden_size: size of hidden state in recurrent neural network.
            emb_model: word embeddings gensim model.
        """
        inputs = [Input(shape=(self.preprocessor.max_len,))]
        emb = [get_emb_layer(self.preprocessor.tokenizer, emb_model=emb_model, trainable=True)(inputs[0])]

        if int(self.mode) > Estimator.Mode.BASELINE:
            inputs.append(Input(shape=(self.preprocessor.max_len,)))
            emb.append(get_emb_layer(self.preprocessor.cue_tokenizer, trainable=True, emb_dim=50)(inputs[1]))

        if int(self.mode) > Estimator.Mode.BASELINE_C:
            inputs.append(Input(shape=(self.preprocessor.max_len,)))
            emb.append(get_emb_layer(self.preprocessor.pos_tokenizer, trainable=True, emb_dim=50)(inputs[2]))

        x = Concatenate()(emb) if len(emb) > 1 else emb[0]
        x = Bidirectional(self.arch.value(hidden_size, return_sequences=True))(x)
        outputs = TimeDistributed(Dense(len(Sentence.Tag), activation='softmax'))(x)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=Adam(lr=lr),
                           loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self, train_data, val_data, epochs=10, tensorboard=False, **kwargs):
        """Trains model.

        Args:
            train_data: list of sentences used for training.
            val_data: list of sentences used for validation.
            epochs: number of epochs in training.
            tensorboard: wether to use tensorboard.
            **kwargs: arguments passed to keras model.fit method.
        """
        train_x, train_y = self.preprocessor.transform(train_data)
        train_x = list(train_x)[:self.mode]

        val_x, val_y = self.preprocessor.transform(val_data)
        val_x = list(val_x)[:self.mode]
        val_data = (val_x, val_y)

        model_tmp = tempfile.NamedTemporaryFile()
        callbacks = [
            EarlyStopping(patience=10),
            ModelCheckpoint(model_tmp.name),
            ReduceLROnPlateau(patience=5),
        ]

        if tensorboard:
            logdir = os.path.join('./logs', self.mode.name, self.arch.name)
            if os.path.exists(logdir) and os.path.isdir(logdir):
                shutil.rmtree(logdir)
            callbacks.append(TensorBoard(logdir))

        self.model.fit(train_x, train_y, validation_data=val_data,
                       callbacks=callbacks, epochs=epochs, **kwargs)

        if epochs:
            self.model.load_weights(model_tmp.name)

    def evaluate(self, sentences, output_file=sys.stdout):
        """Evaluates model.

        Args:
            sentences: sentences to evaulate on.
            output_file: where to output evaluation report.
        """
        data_x, data_y = self.preprocessor.transform(sentences)
        data_x = list(data_x)[:self.mode]

        pred_y = self.model.predict(data_x)
        pred_sentences = self.postprocessor.transform(sentences, pred_y)

        Estimator.tag_score(data_y, pred_y, mask=(data_x[0] != 0), file=output_file)
        starsem_score(map(attrgetter('data'), sentences),
                      map(attrgetter('data'), pred_sentences),
                      file=output_file)

        return pred_sentences

    def save(self, outdir):
        """Saves the estimator, model, preprocessor, postprocessor and mode info."""
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        self.model.save(os.path.join(outdir, 'model.hdf5'))
        with open(os.path.join(outdir, 'meta.pkl'), 'wb') as f:
            pickle.dump({
                'mode': self.mode.name,
                'arch': self.arch.name,
                'preproc': self.preprocessor,
                'postproc': self.postprocessor
            }, f)

    @staticmethod
    def load(model_dir):
        """Loads estimator."""

        with open(os.path.join(model_dir, 'meta.pkl'), 'rb') as f:
            meta_dict = pickle.load(f)

        estimator = Estimator(
            meta_dict['preproc'],
            meta_dict['postproc'],
            mode=meta_dict['mode'],
            arch=meta_dict['arch']
        )

        estimator.model = load_model(os.path.join(model_dir, 'model.hdf5'))

        return estimator
