from keras.layers.embeddings import Embedding
import numpy as np
import gensim


def load_emb_model(embeddings_file):
    """Loads word embeddings model from file."""
    if embeddings_file.endswith('.fasttext.bin'):  # Fasttext binary file
        return gensim.models.FastText.load_fasttext_format(embeddings_file)

    elif embeddings_file.endswith('.bin.gz') or embeddings_file.endswith('.bin'):  # Binary word2vec format
        return gensim.models.KeyedVectors.load_word2vec_format(
            embeddings_file, binary=True, unicode_errors='replace')

    elif embeddings_file.endswith('.txt.gz') or embeddings_file.endswith('.txt') or\
            embeddings_file.endswith('.vec.gz') or embeddings_file.endswith('.vec'):  # Text word2vec format
        return gensim.models.KeyedVectors.load_word2vec_format(
            embeddings_file, binary=False, unicode_errors='replace')

    elif embeddings_file.endswith('.zip'):  # ZIP archive from the NLPL vector repository
        with zipfile.ZipFile(embeddings_file, "r") as archive:
            stream = archive.open("model.txt")
            return gensim.models.KeyedVectors.load_word2vec_format(stream, binary=False, unicode_errors='replace')

    else:  # Native Gensim format?
        return gensim.models.Word2Vec.load(embeddings_file)


def get_emb_layer(tokenizer, emb_model=None, trainable=False, emb_dim=300):
    """Creates Keras embedding layer."""
    num_words = tokenizer.num_words or len(tokenizer.word_index)
    emb_dim = emb_model.wv.vector_size if emb_model else emb_dim

    emb_matrix = np.random.normal(size=(num_words + 1, emb_dim))
    emb_matrix[0] = np.zeros(shape=(emb_dim))

    if emb_model:
        for i in range(1, num_words + 1):
            word = tokenizer.index_word[i]
            try:
                emb_matrix[i] = emb_model.wv.get_vector(word)
            except KeyError:
                pass

    return Embedding(num_words + 1, emb_dim, weights=[emb_matrix], trainable=trainable)
