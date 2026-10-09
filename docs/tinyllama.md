# TinyLlama architecture

DistEngine implements `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, a decoder-only
Transformer that predicts the next token from the tokens before it. It loads
pretrained weights; this project does not train the model.

The configuration below comes from the project's
[pinned checkpoint](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/blob/fe8a4ea1ffedaf415f4da2f062534de366a451e6/config.json).

| Setting | Value |
| --- | --- |
| Vocabulary size | 32,000 |
| Hidden size | 2,048 |
| Decoder blocks | 22 |
| Query heads | 32 |
| Key/value heads | 4 |
| Head dimension | 64 |
| FFN intermediate size | 5,632 |
| Maximum context | 2,048 tokens |
| RMSNorm epsilon | `1e-5` |
| RoPE base | 10,000 |

## From tokens to logits

Here, `B` means batch size and `T` means sequence length.

```text
Token IDs [B, T]
    → embedding [B, T, 2048]
    → 22 decoder blocks [B, T, 2048]
    → final RMSNorm [B, T, 2048]
    → LM head: logits [B, T, 32000]
    → select the last position and choose the next token
```

The tokenizer converts text into token IDs. The embedding looks up one learned
vector for each ID. The LM head maps the final hidden vectors to vocabulary
scores, called logits. Embedding and LM-head weights are separate.

## Inside a decoder block

Each block first applies attention, then a feed-forward network (FFN):

```text
h = x + Attention(RMSNorm(x))
y = h + FFN(RMSNorm(h))
```

RMSNorm divides each token vector by its root mean square, with epsilon for
stability, then applies a learned feature scale. Residual additions preserve the
incoming representation while adding the result of each branch.

Attention projects hidden states into queries (Q), keys (K), and values (V).
Query/key dot products, scaled by `1 / sqrt(64)`, measure attention scores.
A causal mask prevents a token from attending to future tokens. Softmax turns
scores into probabilities, which weight the value vectors. The heads are then
combined and projected back to the hidden size.

## GQA and RoPE

Grouped-query attention (GQA) shares each of the four KV heads among eight query
heads. Before expansion for the current attention computation, the shapes are:

| Tensor | Shape |
| --- | --- |
| Q | `[B, 32, T, 64]` |
| K | `[B, 4, T, 64]` |
| V | `[B, 4, T, 64]` |
| Attention scores | `[B, 32, T, T]` |

Rotary positional embedding (RoPE) rotates pairs of features in Q and K according
to their absolute token positions. It supplies positional information without
adding a position vector to the embedding. V is not rotated. DistEngine uses
the half-split convention: feature 0 pairs with 32, feature 1 with 33, and so on.

## Feed-forward network

The FFN uses SwiGLU:

```text
FFN(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))
SiLU(z) = z * sigmoid(z)
```

Gate and up projections each produce `[B, T, 5632]`. Their elementwise product
is projected back to `[B, T, 2048]`. The FFN processes each token independently;
attention is what mixes information across token positions.

## Generation and the future KV cache

Greedy generation selects `argmax` from the final position's logits, appends that
token, and repeats until EOS or the output limit. The current implementation
recomputes the whole sequence at every step and supports one unpadded request.

A future KV cache will retain each layer's already-computed keys and values:

```text
K cache per layer: [B, 4, cached_length, 64]  (after RoPE)
V cache per layer: [B, 4, cached_length, 64]
```

Prefill processes the prompt and fills those caches. Decode processes one new
token at its absolute position and attends to cached K/V. Cache storage keeps
four KV heads, rather than the 32 heads expanded for attention. KV caching is
planned for stage 6 and is not implemented yet.

## Source files

| File under `src/distengine/` | Responsibility |
| --- | --- |
| `layers/embed_head.py` | Token lookup and vocabulary projection |
| `layers/linear.py` | Learned linear projections |
| `layers/layernorm.py` | RMSNorm |
| `layers/activation.py` | SiLU and gate/up multiplication |
| `layers/rotary_embedding.py` | Position-dependent Q/K rotation |
| `layers/attention.py` | GQA, causal masking, and attention projections |
| `models/llama.py` | FFN, decoder blocks, and complete model |
| `load_model.py` | Copy pinned HF checkpoint weights into custom modules |
| `layers/sampler.py` | Greedy token selection |
| `generation.py` | Manual generation loop |

See the [README](../README.md#run-the-custom-tinyllama-model) for the run command.
