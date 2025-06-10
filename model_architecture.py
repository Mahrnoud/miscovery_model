import torch
import math
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import Seq2SeqLMOutput


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normalized = x / rms
        return self.weight * x_normalized


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-4):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return self.gamma * x


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=1024, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self.max_seq_len_cached = max_position_embeddings
        self._update_cos_sin_cache(max_position_embeddings)

    def _update_cos_sin_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]

        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            self._update_cos_sin_cache(seq_len)

        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...]
        )


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_embeddings(q, k, cos, sin):
    q_seq_len = q.shape[2]
    k_seq_len = k.shape[2]

    q_cos = cos[:, :, :q_seq_len, :]
    q_sin = sin[:, :, :q_seq_len, :]
    k_cos = cos[:, :, :k_seq_len, :]
    k_sin = sin[:, :, :k_seq_len, :]

    q_embed = (q * q_cos) + (rotate_half(q) * q_sin)
    k_embed = (k * k_cos) + (rotate_half(k) * k_sin)

    return q_embed, k_embed


try:
    from flash_attn import flash_attn_func

    FLASH_ATTENTION_AVAILABLE = True
    print("Flash Attention is available and will be used for faster training")
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    print("Flash Attention not available, falling back to standard attention")


def create_decoder_mask(tgt, pad_idx):
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    tgt_len = tgt.size(1)
    tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
    tgt_mask = tgt_pad_mask & tgt_sub_mask.unsqueeze(0)
    return tgt_mask


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, max_len=1024, use_flash_attn=False):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.use_flash_attn = use_flash_attn and FLASH_ATTENTION_AVAILABLE

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.depth = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Increased dropout rate
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        # Layer scaling for better gradient flow
        self.layer_scale = LayerScale(d_model, init_values=1e-4)

        self.rotary_emb = RotaryEmbedding(self.depth, max_position_embeddings=max_len)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1, 2)

    def forward(self, query, key=None, value=None, mask=None):
        batch_size = query.shape[0]

        if key is None:
            key = query
        if value is None:
            value = query

        q = self.split_heads(self.q_proj(query), batch_size)
        k = self.split_heads(self.k_proj(key), batch_size)
        v = self.split_heads(self.v_proj(value), batch_size)

        max_seq_len = max(q.shape[2], k.shape[2])
        cos, sin = self.rotary_emb(q, seq_len=max_seq_len)
        q, k = apply_rotary_embeddings(q, k, cos, sin)

        if self.use_flash_attn and q.is_cuda:
            try:
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)

                if mask is not None:
                    attn_mask = mask.float()
                    attn_mask = (1.0 - attn_mask) * -10000.0
                    if len(attn_mask.shape) == 4:
                        attn_mask = attn_mask.squeeze(1)
                else:
                    attn_mask = None

                attn_output = flash_attn_func(
                    q, k, v,
                    dropout_p=self.attention_dropout.p,
                    attn_mask=attn_mask,
                    causal=False
                )

                if attn_output.shape[1] != self.num_heads:
                    attn_output = attn_output.transpose(1, 2)
            except Exception as e:
                print(f"Flash attention error: {e}. Falling back to standard attention.")
                # Reshape back for standard attention
                if q.shape[1] != self.num_heads:
                    q = q.transpose(1, 2)
                    k = k.transpose(1, 2)
                    v = v.transpose(1, 2)

                # Use standard attention
                attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.depth)
                if mask is not None:
                    attn_logits = attn_logits.masked_fill(~mask, float('-inf'))
                attn_weights = self.attention_dropout(torch.nn.functional.softmax(attn_logits, dim=-1))
                attn_output = torch.matmul(attn_weights, v)
        else:
            attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.depth)
            if mask is not None:
                attn_logits = attn_logits.masked_fill(~mask, float('-inf'))
            attn_weights = self.attention_dropout(torch.nn.functional.softmax(attn_logits, dim=-1))
            attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(attn_output)
        # Apply layer scaling and dropout
        output = self.layer_scale(output)
        return self.output_dropout(output)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super(FeedForward, self).__init__()
        if d_ff is None:
            # Increase to 4x for better representation capacity
            d_ff = 4 * d_model

        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_scale = LayerScale(d_model, init_values=1e-4)

    def forward(self, x):
        # SwiGLU-like activation
        gated_output = self.w1(x) * torch.sigmoid(self.w2(x) * 1.0)
        output = self.w3(self.dropout(gated_output))
        # Apply layer scaling
        return self.layer_scale(output)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-norm architecture
        attn_input = self.norm1(x)
        attn_output = self.self_attention(attn_input, mask=mask)
        x = x + self.dropout1(attn_output)

        ff_input = self.norm2(x)
        ff_output = self.feed_forward(ff_input)
        x = x + self.dropout2(ff_output)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size, max_len, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Scale embeddings
        self.embed_scale = math.sqrt(d_model)
        # Dropout for embeddings
        self.embed_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(self, x, mask=None):
        # Apply embedding with scaling but without positional encoding (handled by RoPE)
        x = self.embedding(x) * self.embed_scale
        x = self.embed_dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Pre-norm architecture
        attn_input = self.norm1(x)
        self_attn_output = self.self_attention(attn_input, mask=tgt_mask)
        x = x + self.dropout1(self_attn_output)

        cross_attn_input = self.norm2(x)
        cross_attn_output = self.cross_attention(
            query=cross_attn_input,
            key=enc_output,
            value=enc_output,
            mask=src_mask
        )
        x = x + self.dropout2(cross_attn_output)

        ff_input = self.norm3(x)
        ff_output = self.feed_forward(ff_input)
        x = x + self.dropout3(ff_output)

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size, max_len, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Scale embeddings
        self.embed_scale = math.sqrt(d_model)
        # Dropout for embeddings
        self.embed_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Apply embedding with scaling but without positional encoding (handled by RoPE)
        x = self.embedding(x) * self.embed_scale
        x = self.embed_dropout(x)

        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)

        x = self.norm(x)
        return self.output_projection(x)


class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_encoder_layers, num_decoder_layers,
                 vocab_size, max_len, pad_idx, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, num_encoder_layers,
                                          vocab_size, max_len, dropout)
        self.decoder = TransformerDecoder(d_model, num_heads, d_ff, num_decoder_layers,
                                          vocab_size, max_len, dropout)
        self.pad_idx = pad_idx

    def create_masks(self, src, tgt):
        if not isinstance(src, torch.Tensor):
            src = torch.tensor(src)
        if not isinstance(tgt, torch.Tensor):
            tgt = torch.tensor(tgt)

        src_pad_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.size(1)
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        tgt_mask = tgt_pad_mask & tgt_sub_mask.unsqueeze(0)

        return src_pad_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.create_masks(src, tgt)
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        return dec_output


class CustomTransformerConfig(PretrainedConfig):
    model_type = "miscovery"

    def __init__(
            self,
            vocab_size=100000,
            d_model=384,
            num_heads=8,
            d_ff=1536,
            num_encoder_layers=8,
            num_decoder_layers=8,
            max_position_embeddings=1024,
            dropout=0.1,
            pad_token_id=0,
            bos_token_id=2,
            eos_token_id=3,
            use_flash_attn=True,
            **kwargs
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.max_position_embeddings = max_position_embeddings
        self.dropout = dropout
        self.use_flash_attn = use_flash_attn

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )


class CustomTransformerModel(PreTrainedModel, GenerationMixin):
    config_class = CustomTransformerConfig
    main_input_name = "input_ids"

    def __init__(self, config):
        super().__init__(config)
        self.model = Transformer(
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            vocab_size=config.vocab_size,
            max_len=config.max_position_embeddings,
            pad_idx=config.pad_token_id,
            dropout=config.dropout
        )

        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        self.generate_response = self.custom_generate.__get__(self)
        self.encoder.main_input_name = "input_ids"

    def forward(
            self,
            input_ids=None,
            decoder_input_ids=None,
            attention_mask=None,
            decoder_attention_mask=None,
            labels=None,
            label_smoothing=0.1,  # Added label smoothing
            **kwargs
    ):
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)
        elif decoder_input_ids is None:
            decoder_input_ids = input_ids

        outputs = self.model(src=input_ids, tgt=decoder_input_ids)

        loss = None
        if labels is not None:
            # Use label smoothing if specified
            if label_smoothing > 0:
                loss_fct = nn.CrossEntropyLoss(
                    ignore_index=self.config.pad_token_id,
                    label_smoothing=label_smoothing
                )
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)

            shifted_logits = outputs[:, :-1, :].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            loss = loss_fct(shifted_logits.view(-1, self.config.vocab_size), shifted_labels.view(-1))

        return Seq2SeqLMOutput(
            loss=loss,
            logits=outputs,
        )

    def _shift_right(self, input_ids):
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = self.config.bos_token_id
        return shifted_input_ids

    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past_key_values=None,
            attention_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs
    ):
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        if encoder_outputs is None and kwargs.get("input_ids") is not None:
            input_ids = kwargs.get("input_ids")
            src_pad_mask = (input_ids != self.config.pad_token_id).unsqueeze(1).unsqueeze(2)
            encoder_outputs = self.model.encoder(input_ids, src_pad_mask)

        return {
            "decoder_input_ids": decoder_input_ids,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "past_key_values": past_key_values
        }

    def custom_generate(
            self,
            prompt,
            tokenizer,
            max_length=256,
            device='cuda',
            temperature=1.0,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.0,
            do_sample=True
    ):
        """
        Enhanced text generation with sampling options
        """
        self.eval()

        input_ids = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=256,
            padding='max_length',
            truncation=True
        )["input_ids"].to(device)

        bos_token_id = tokenizer.bos_token_id
        if bos_token_id is None:
            if tokenizer.cls_token_id is not None:
                bos_token_id = tokenizer.cls_token_id
            else:
                bos_token_id = 1  # Fallback

        decoder_input = torch.tensor([[bos_token_id]], device=device)
        generated_tokens = [bos_token_id]

        # Stop tokens to check
        stop_tokens = []
        if tokenizer.eos_token_id is not None:
            stop_tokens.append(tokenizer.eos_token_id)
        if tokenizer.sep_token_id is not None:
            stop_tokens.append(tokenizer.sep_token_id)

        for _ in range(max_length):
            src_mask, tgt_mask = self.model.create_masks(input_ids, decoder_input)
            enc_output = self.model.encoder(input_ids, src_mask)
            dec_output = self.model.decoder(decoder_input, enc_output, src_mask, tgt_mask)

            next_token_logits = dec_output[:, -1, :].squeeze(0)

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for prev_token in generated_tokens:
                    next_token_logits[prev_token] /= repetition_penalty

            # Filter with top-k
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Filter with top-p (nucleus sampling)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p

                # Shift the indices to the right to keep the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample or greedy selection
            if do_sample:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

            next_token = next_token.unsqueeze(0)
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            generated_tokens.append(next_token.item())

            # Check stop condition
            if next_token.item() in stop_tokens:
                break

            # Check for repetition
            if len(generated_tokens) >= 4:
                # Stop if generating the same token 4 times in a row
                if len(set(generated_tokens[-4:])) == 1:
                    break

        output_text = tokenizer.decode(decoder_input[0].tolist(), skip_special_tokens=True)
        return output_text
