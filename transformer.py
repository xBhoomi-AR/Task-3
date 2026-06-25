import math
import torch
from torch import nn

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
torch.cuda.empty_cache()

num_heads = 8
embed_len = 512
batch_size = 8              
stack_len = 6               
dropout = 0.1               

output_vocab_size = 7000
input_vocab_size = 7000  

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            0,
            max_len,
            dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            ).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ScaledDotProduct(nn.Module):

    def __init__(self, embed_len=embed_len, mask=False):
        super(ScaledDotProduct, self).__init__()

        self.dk = embed_len
        self.mask = mask
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries, keys, values):

        # queries: (batch, heads, seq_len, head_dim)
        # keys:    (batch, heads, seq_len, head_dim)
        # values:  (batch, heads, seq_len, head_dim)

        compatibility = torch.matmul(
            queries,
            keys.transpose(-2, -1)
        )

        compatibility = compatibility / math.sqrt(
            queries.size(-1)
        )

        # Causal mask for decoder self-attention
        if self.mask:

            seq_len = compatibility.size(-1)

            mask = torch.triu(
                torch.ones(
                    seq_len,
                    seq_len,
                    device=compatibility.device
                ),
                diagonal=1
            ).bool()

            compatibility = compatibility.masked_fill(
                mask,
                float("-inf")
            )

        attention_weights = self.softmax(
            compatibility
        )

        output = torch.matmul(
            attention_weights,
            values
        )

        return output

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        num_heads=num_heads,
        embed_len=embed_len,
        mask=None
    ):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.embed_len = embed_len
        self.head_length = embed_len // num_heads
        self.mask = mask

        # Q, K, V projections
        self.q_linear = nn.Linear(embed_len, embed_len)
        self.k_linear = nn.Linear(embed_len, embed_len)
        self.v_linear = nn.Linear(embed_len, embed_len)

        # Attention
        if self.mask:
            self.attention = ScaledDotProduct(mask=True)
        else:
            self.attention = ScaledDotProduct()

        self.output_linear = nn.Linear(embed_len, embed_len)

    def forward(self, queries, keys, values):

        batch_size = queries.size(0)

        # Q
        queries = self.q_linear(queries).view(
            batch_size,
            -1,
            self.num_heads,
            self.head_length
        )
        queries = queries.transpose(1, 2)

        # K
        keys = self.k_linear(keys).view(
            batch_size,
            -1,
            self.num_heads,
            self.head_length
        )
        keys = keys.transpose(1, 2)

        # V
        values = self.v_linear(values).view(
            batch_size,
            -1,
            self.num_heads,
            self.head_length
        )
        values = values.transpose(1, 2)

        # Attention output
        sdp_output = self.attention(
            queries,
            keys,
            values
        )

        # Concatenate heads
        sdp_output = sdp_output.transpose(1, 2).contiguous().view(
            batch_size,
            -1,
            self.embed_len
        )

        return self.output_linear(sdp_output)

class EncoderBlock(nn.Module):

    def __init__(
        self,
        embed_len=embed_len,
        dropout=dropout
    ):
        super().__init__()

        self.multihead = MultiHeadAttention()

        self.norm1 = nn.LayerNorm(embed_len)
        self.norm2 = nn.LayerNorm(embed_len)

        self.dropout = nn.Dropout(dropout)

        self.feedForward = nn.Sequential(
            nn.Linear(embed_len, embed_len * 4),
            nn.ReLU(),
            nn.Linear(embed_len * 4, embed_len)
        )

    def forward(self, x):

        attention = self.multihead(
            x,
            x,
            x
        )

        x = self.norm1(
            x + self.dropout(attention)
        )

        ff = self.feedForward(x)

        x = self.norm2(
            x + self.dropout(ff)
        )

        return x

class DecoderBlock(nn.Module):

    def __init__(
        self,
        embed_len=embed_len,
        dropout=dropout
    ):
        super().__init__()

        # Masked Self Attention
        self.masked_attention = MultiHeadAttention(mask=True)

        # Encoder-Decoder Attention
        self.cross_attention = MultiHeadAttention()

        self.norm1 = nn.LayerNorm(embed_len)
        self.norm2 = nn.LayerNorm(embed_len)
        self.norm3 = nn.LayerNorm(embed_len)

        self.dropout = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_len, embed_len * 4),
            nn.ReLU(),
            nn.Linear(embed_len * 4, embed_len)
        )

    def forward(
        self,
        x,
        encoder_output
    ):

        # ---------------------------
        # 1. Masked Self Attention
        # ---------------------------
        masked_output = self.masked_attention(
            x,
            x,
            x
        )

        x = self.norm1(
            x + self.dropout(masked_output)
        )

        # ---------------------------
        # 2. Encoder-Decoder Attention
        # ---------------------------
        cross_output = self.cross_attention(
            x,
            encoder_output,
            encoder_output
        )

        x = self.norm2(
            x + self.dropout(cross_output)
        )

        # ---------------------------
        # 3. Feed Forward
        # ---------------------------
        ff_output = self.feed_forward(x)

        x = self.norm3(
            x + self.dropout(ff_output)
        )

        return x        
class Transformer(nn.Module):

    def __init__(
        self,
        stack_len=stack_len,
        embed_len=embed_len,
        device=device,
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size
    ):
        super(Transformer, self).__init__()

        self.stack_len = stack_len
        self.embed_len = embed_len
        self.device = device

        # Token embeddings
        self.src_embedding = nn.Embedding(
            input_vocab_size,
            embed_len
        )

        self.tgt_embedding = nn.Embedding(
            output_vocab_size,
            embed_len
        )

        # Sinusoidal positional encoding
        self.positional_encoding = PositionalEncoding(
            embed_len
        )

        # Encoder stack
        self.encStack = nn.ModuleList(
            [EncoderBlock() for _ in range(stack_len)]
        )

        # Decoder stack
        self.decStack = nn.ModuleList(
            [DecoderBlock() for _ in range(stack_len)]
        )

        # Final output layer
        self.finalLinear = nn.Linear(
            embed_len,
            output_vocab_size
        )

    def forward(self, src, tgt):

    # ======================
    # Source Embedding
    # ======================
      src = self.src_embedding(src)
      src = self.positional_encoding(src)

    # ======================
    # Target Embedding
    # ======================
      tgt = self.tgt_embedding(tgt)
      tgt = self.positional_encoding(tgt)

    # ======================
    # Encoder
    # ======================
      enc_output = src

      for enc_layer in self.encStack:
          enc_output = enc_layer(enc_output)

    # ======================
    # Decoder
    # ======================
      dec_output = tgt

      for dec_layer in self.decStack:
          dec_output = dec_layer(
              dec_output,
              enc_output
          )

    # ======================
    # Output Projection
    # ======================
      output = self.finalLinear(dec_output)

      return output