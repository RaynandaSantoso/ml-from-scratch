import numpy as np

np.random.seed(42)

vocab = {
    "the": 0,
    "cat": 1,
    "sleeps": 2
}

sentence = "cat the cat"
tokens = sentence.split(" ")
token_ids = [vocab[token] for token in tokens]

# create initial random matrix.
embedding_matrix = np.random.randn(3, 4)
sentence_embeddings = embedding_matrix[token_ids]
positions = np.arange(len(token_ids))

# another random matrix.
position_matrix = np.random.randn(3,4)
positional_embeddings = position_matrix[positions]

final_embeddings = sentence_embeddings + positional_embeddings

if __name__ == "__main__":    
    print("sentence", sentence_embeddings)
    print("positional", positional_embeddings)
    print("sentence + positional embedding", final_embeddings)