# INF5820-Assignment4
Fourth and final Obligatory exercise.

## Training
To train the models use `train.py` script:
```
usage: train.py [-h] [--training TRAINING] [--validation VALIDATION]
                [--metrics METRICS]
                [--mode {BASELINE,BASELINE_C,BASELINE_C_POS}]
                [--arch {LSTM,GRU}] [--word_vectors WORD_VECTORS]
                [--epochs EPOCHS] [--lr LR] [--max_len MAX_LEN]
                [--hidden_size HIDDEN_SIZE] [--negations_only] [--tensorboard]
                [--save SAVE]

optional arguments:
  -h, --help            show this help message and exit
  --training TRAINING   epe file with training sentences.
  --validation VALIDATION
                        epe file with validation sentences.
  --metrics METRICS     where to output metrics computed on validation.
  --mode {BASELINE,BASELINE_C,BASELINE_C_POS}
                        mode in which model is trained, C adds cue
                        information, POS adds pos tags.
  --arch {LSTM,GRU}     which recurrent network architecture to use.
  --word_vectors WORD_VECTORS
                        file with word embeddings
  --epochs EPOCHS       number of epochs to train the model.
  --lr LR               learning rate used in adam optimizer.
  --max_len MAX_LEN     maximum length of sentence.
  --hidden_size HIDDEN_SIZE
                        memory size of recurrent neural network.
  --negations_only      if present, only sentences with at least one negation
                        are used.
  --tensorboard
  --save SAVE           directory where to save model.
```

## Evaluation:
Evaluate saved models with `eval.py`:
```
usage: eval.py [-h] [--sentences SENTENCES] [--metrics METRICS]
               [--negations_only]
               model_dir

positional arguments:
  model_dir             directory where the model is saved

optional arguments:
  -h, --help            show this help message and exit
  --sentences SENTENCES
                        epe file with data to evaluate on
  --metrics METRICS     file where to output computed metrics.
  --negations_only      use only sentences with at least one negation.
```

Evaluate baeline model with:
```
python ./eval.py ./baseline_model/ --sentences [SENTENCES]
```

Evaluate best model with:
```
python ./eval.py ./best_model/ --sentences [SENTENCES]
```