import torch
from torch import nn
from transformers import LlamaConfig

from layers.embed_head import LMHead, TokenEmbedding
from layers.activation import SiluAndMul
from layers.attention import LlamaAttention
from layers.layernorm import RMSNorm
from layers.linear import Linear
from distengine.paged_kv_cache import PagedKVCache, BatchedPagedKVCache


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()

        self.gate_proj = Linear(hidden_size, intermediate_size)
        self.up_proj = Linear(hidden_size, intermediate_size)
        self.down_proj = Linear(intermediate_size, hidden_size)
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        packed = torch.cat((gate, up), dim=-1)
        activated = self.act_fn(packed)

        return self.down_proj(activated)
    
    
class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        rms_norm_eps: float,
        rope_theta: float,
    ):
        super().__init__()

        self.input_layernorm = RMSNorm(
            hidden_size,
            eps=rms_norm_eps,
        )

        self.self_attn = LlamaAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            rope_theta=rope_theta,
        )

        self.post_attention_layernorm = RMSNorm(
            hidden_size,
            eps=rms_norm_eps,
        )

        self.mlp = LlamaMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        cache: PagedKVCache | BatchedPagedKVCache | None = None,
        layer_index: int = 0,
    ) -> torch.Tensor:
        
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(
            x,
            positions,
            cache=cache,
            layer_index=layer_index,
        )
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x
    

class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.embed_tokens = TokenEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
        )

        num_kv_heads = config.num_key_value_heads
        if num_kv_heads is None:
            num_kv_heads = config.num_attention_heads

        rope_parameters = config.rope_parameters
        if rope_parameters is None:
            raise ValueError("Expected RoPE parameters in the model config")
        if rope_parameters.get("rope_type", "default") != "default":
            raise ValueError("Only default RoPE is supported")
        rope_theta = float(rope_parameters["rope_theta"])

        self.layers = nn.ModuleList([
            LlamaDecoderLayer(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=num_kv_heads,
                rms_norm_eps=config.rms_norm_eps,
                rope_theta=rope_theta,
            )
            for _ in range(config.num_hidden_layers)
        ])

        self.norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        cache: PagedKVCache | BatchedPagedKVCache | None = None,
    ) -> torch.Tensor:
        
        batch_size, sequence_length = input_ids.shape

        past_length = 0

        batched_cache = isinstance(cache, BatchedPagedKVCache)
        if cache is not None and not batched_cache:
            if batch_size != 1:
                raise ValueError("Cached inference supports batch size one")

            if cache.num_layers != len(self.layers):
                raise ValueError("Cache layer count does not match the model")

            past_length = cache.sequence_length

            if past_length + sequence_length > cache.capacity:
                raise ValueError("KV cache capacity exceeded")

        expected_positions = torch.arange(
            past_length,
            past_length + sequence_length,
            device=input_ids.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(batch_size, -1)

        if batched_cache:
            if sequence_length != 1 or batch_size != len(cache.caches) or cache.num_layers != len(self.layers):
                raise ValueError("Invalid decode batch shape or layer count")
            expected_positions = torch.tensor(cache.lengths, device=input_ids.device).unsqueeze(1)

        if positions is None:
            positions = expected_positions
        elif cache is not None:
            if not torch.equal(positions, expected_positions):
                raise ValueError("Cached positions must continue from the cache length")

        x = self.embed_tokens(input_ids)

        for layer_index, layer in enumerate(self.layers):
            x = layer(
                x,
                positions,
                cache=cache,
                layer_index=layer_index,
            )

        return self.norm(x)
    

class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.config = config
        self.model = LlamaModel(config)

        self.lm_head = LMHead(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        cache: PagedKVCache | BatchedPagedKVCache | None = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            cache=cache,
        )
        return self.lm_head(hidden_states)
