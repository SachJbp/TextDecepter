import numpy as np

def pick_most_similar_words_batch(src_words, sim_mat, idx2word, ret_count=10, threshold=0.):
    """
    embeddings is a matrix with (d, vocab_size)
    """
    sim_order = np.argsort(-sim_mat[src_words, :])[:, 1:1 + ret_count]
    sim_words, sim_values = [], []
    for idx, src_word in enumerate(src_words):
        sim_value = sim_mat[src_word][sim_order[idx]]
        mask = sim_value >= threshold
        sim_word, sim_value = sim_order[idx][mask], sim_value[mask]
        sim_word = [idx2word[id] for id in sim_word]
        sim_words.append(sim_word)
        sim_values.append(sim_value)
    return sim_words, sim_values
