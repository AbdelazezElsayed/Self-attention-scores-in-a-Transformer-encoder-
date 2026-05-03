import numpy as np

# Sentence
sentence = "What are the symptoms of diabetes"
tokens = sentence.lower().split()

# Dummy embeddings (seq_len, embed_dim)
X = np.array([
    [0.20, 0.10, 0.40, 0.30],  # what
    [0.30, 0.20, 0.10, 0.50],  # are
    [0.40, 0.30, 0.20, 0.10],  # the
    [0.90, 0.70, 0.30, 0.20],  # symptoms
    [0.40, 0.30, 0.60, 0.20],  # of
    [0.80, 0.90, 0.50, 0.40],  # diabetes
])

seq_len, embed_dim = X.shape
num_heads = 2
head_dim = embed_dim // num_heads

# Random weight matrices (like learned parameters)
np.random.seed(42)
Wq = np.random.randn(embed_dim, embed_dim)
Wk = np.random.randn(embed_dim, embed_dim)
Wv = np.random.randn(embed_dim, embed_dim)
Wo = np.random.randn(embed_dim, embed_dim)


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def split_heads(x):
    # (seq_len, embed_dim) → (num_heads, seq_len, head_dim)
    return x.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)


def combine_heads(x):
    # (num_heads, seq_len, head_dim) → (seq_len, embed_dim)
    return x.transpose(1, 0, 2).reshape(seq_len, embed_dim)


# Step 1: Linear projections
Q = X @ Wq
K = X @ Wk
V = X @ Wv

# Step 2: Split into heads
Q = split_heads(Q)
K = split_heads(K)
V = split_heads(V)

# Step 3: Scaled dot-product attention
scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(head_dim)
attention_weights = softmax(scores)

# Step 4: Apply attention to values
context = np.matmul(attention_weights, V)

# Step 5: Combine heads
context = combine_heads(context)

# Step 6: Final linear projection
output = context @ Wo

# Print results
print("Tokens:")
print(tokens)

print("\nAttention Weights Shape:", attention_weights.shape)
print("Attention Weights:")
print(attention_weights)

print("\nFinal Output Shape:", output.shape)
print("Final Output:")
print(output)