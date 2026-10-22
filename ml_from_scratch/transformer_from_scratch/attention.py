import numpy as np
from ml_from_scratch.transformer_from_scratch.embeddings import final_embeddings
from ml_from_scratch.activations import softmax

np.random.seed(42)

W_Q = np.random.randn(4,2)
W_K = np.random.randn(4,2)
W_V = np.random.randn(4,2)

X = final_embeddings

# Q K V 
Q = X @ W_Q
K = X @ W_K
V = X @ W_V

K_T = np.transpose(K)
#Score
scores = Q @ K.T

# calculating Attention weights:
# Softmax (Q_KT / sqrt dimension)

d_k = K.shape[-1]
scaled_scores = scores / np.sqrt(d_k)

# mask unrelated to neg infinity
seq_len = scores.shape[0]

mask = np.triu(
    np.ones((seq_len, seq_len), dtype=bool),
    k=1
)

masked_scores = np.where(mask, -np.inf, scaled_scores)

attention_weights = softmax(masked_scores)
attention_output = attention_weights @ V
print(attention_output)