import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, num_heads, embedding_size):
        super(SelfAttention, self).__init__()
        self.attention_size = embedding_size // num_heads # = AS
        self.W_q = nn.Linear(embedding_size, self.attention_size, bias=False)
        self.W_k = nn.Linear(embedding_size, self.attention_size, bias=False)
        self.W_v = nn.Linear(embedding_size, self.attention_size, bias=False)

    def forward(self, queries, keys, values, mask=None):
        #queries, keys, values : (B, L, E), mask: (B, L, L)
        Q = self.W_q(queries) # (B, L, AS)
        K = self.W_k(keys) # (B, L, AS)
        V = self.W_v(values) # (B, L, AS)
        norm_factor = math.sqrt(self.attention_size)
        #in decoder Q: (B, L_trg, AS), K: (B, L_src, AS)
        dot_products = torch.bmm(Q, K.permute(0, 2, 1)) / norm_factor

        if mask is not None:
            dot_products = dot_products.masked_fill(mask, -1e20) # (B, L, L)

        attention_score = nn.functional.softmax(dot_products, dim=2) # (B, L, L)
        attention = torch.bmm(attention_score, V) # (B, L, AS)

        #(B, L, AS)
        return attention

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embedding_size):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads #= NH
        self.embedding_size = embedding_size
        self.attention_heads = nn.ModuleList([SelfAttention(num_heads, embedding_size) for _ in range(num_heads)])
        self.linear = nn.Linear(embedding_size, embedding_size, bias=False)

    def forward(self, queries, keys, values, mask=None):
        #queries, keys, values : (B, L, E), mask: (B, L, L)
        attentions = list()
        for head in self.attention_heads:
            attention = head(queries, keys, values, mask)
            attentions += [attention]

        #attentions[i] : (B, L, AS), AS * NH = E
        attentions = torch.cat(attentions, dim=-1)

        output = self.linear(attentions)
        #output: (B, L, E)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, embedding_size, prob):
        super().__init__()
        self.pos_features = torch.zeros(max_len, embedding_size)

        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        freqs = torch.exp(torch.arange(0, embedding_size, 2, dtype=torch.float) * \
                          (-math.log(10000) / embedding_size)).unsqueeze(0)

        arguments = positions * freqs
        self.pos_features[:, 0::2] = torch.sin(arguments)
        self.pos_features[:, 1::2] = torch.cos(arguments)
        self.pos_features = self.pos_features.unsqueeze(0)
        self.pos_features = nn.Parameter(self.pos_features, requires_grad=False)
        # pos_features: (1, max_length, embed_dim)

        self.dropout = nn.Dropout(p=prob)

    def forward(self, inputs):
        outputs = inputs + self.pos_features[:, :inputs.shape[1]]
        return self.dropout(outputs)

class FeedForward(nn.Module):
    def __init__(self, embedding_size, feed_forward_size):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(embedding_size, feed_forward_size)
        self.linear_2 = nn.Linear(feed_forward_size, embedding_size)
        self.relu = nn.ReLU()

    def forward(self, input):
        #input (B, L, E)
        output = self.relu(self.linear_1(input))
        #output: (B, L, E)
        return self.linear_2(output)

class ResConnection(nn.Module):
    def __init__(self, embedding_size):
        super(ResConnection, self).__init__()
        self.layer_norm = nn.LayerNorm(embedding_size)

    def forward(self, input, layer):
        #input : (B, L, E)
        #output: (B, L, E)
        return input + layer(self.layer_norm(input))

class EncoderBlock(nn.Module):
    def __init__(self, num_heads, embedding_size, feed_forward_size):
        super(EncoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(num_heads, embedding_size)
        self.feed_forward = FeedForward(embedding_size, feed_forward_size)
        self.res_connection_1 = ResConnection(embedding_size)
        self.res_connection_2 = ResConnection(embedding_size)

    def forward(self, input, mask=None):
        output = self.res_connection_1(input, lambda x: self.self_attention(x, x, x, mask))
        return self.res_connection_2(output, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, num_blocks, num_tokens, embedding_size, num_heads, feed_forward_size, max_len, prob):
        super(Encoder, self).__init__()
        self.num_blocks = num_blocks
        self.num_tokens = num_tokens
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.feed_forward_size = feed_forward_size
        self.max_len = max_len

        self.positional_encoding = PositionalEncoding(max_len, embedding_size, prob)
        self.embedding = nn.Embedding(num_tokens, embedding_size, padding_idx=3)
        self.blocks = nn.ModuleList([EncoderBlock(num_heads, embedding_size, feed_forward_size) for _ in range(num_blocks)])
        self.layer_norm = nn.LayerNorm(embedding_size)
        
    def forward(self, input, mask=None):
        #input: (B, L)
        embedding = self.embedding(input)
        output = self.positional_encoding(embedding)
        for encoder_block in self.blocks:
            output = encoder_block(output, mask)
        #output: (B, L, E)
        return self.layer_norm(output)

class DecoderBlock(nn.Module):
    def __init__(self, num_heads, embedding_size, feed_forward_size):
        super(DecoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(num_heads, embedding_size)
        self.attention = MultiHeadAttention(num_heads, embedding_size)
        self.feed_forward = FeedForward(embedding_size, feed_forward_size)
        self.res_connection_1 = ResConnection(embedding_size)
        self.res_connection_2 = ResConnection(embedding_size)
        self.res_connection_3 = ResConnection(embedding_size)

    def forward(self, input, encoder_output, mask=None):
        #input : (B, L_out, E), encoder_output: (B, L_in, E)
        output = self.res_connection_1(input, lambda x: self.self_attention(x, x, x, mask))
        output = self.res_connection_2(output, lambda x: self.attention(x, encoder_output, encoder_output))
        #output: (B, L_out, E)
        return self.res_connection_3(output, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self, num_blocks, num_tokens, embedding_size, num_heads, feed_forward_size, max_len, prob):
        super(Decoder, self).__init__()
        self.num_blocks = num_blocks
        self.num_tokens = num_tokens
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.feed_forward_size = feed_forward_size
        self.max_len = max_len

        self.positional_encoding = PositionalEncoding(max_len, embedding_size, prob)
        self.embedding = nn.Embedding(num_tokens, embedding_size, padding_idx=3)
        self.blocks = nn.ModuleList([DecoderBlock(num_heads, embedding_size, feed_forward_size) for _ in range(num_blocks)])
        self.linear = nn.Linear(embedding_size, num_tokens)

    def forward(self, input, encoder_output, mask=None):
        #input : (B, L_out), encoder_output: (B, L_in, E)
        embedding = self.embedding(input)
        output = self.positional_encoding(embedding)
        for decoder_block in self.blocks:
            output = decoder_block(output, encoder_output, mask)
        #returning output: (B, L_out, NT)
        return self.linear(output)

class Transformer(nn.Module):
    def __init__(self, num_blocks, num_tokens_src, num_tokens_trg, embedding_size, num_heads, feed_forward_size, max_len, device, prob):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_blocks, num_tokens_src, embedding_size, num_heads, feed_forward_size, max_len, prob)
        self.decoder = Decoder(num_blocks, num_tokens_trg, embedding_size, num_heads, feed_forward_size, max_len, prob)
        self.device = device

    def create_mask(self, input):
    #target: (B, L_out, E)
        batch_size, length = input.shape[0], input.shape[1]
        mask = torch.tril(torch.ones((length, length)))
        mask = mask.expand(batch_size, length, length)
        mask = (1. - mask).bool()
        #output: (B, L, L)
        return mask.to(self.device)

    def forward(self, src, trg):
        encoder_output = self.encoder(src)
        trg_mask = self.create_mask(trg)
        return self.decoder(trg, encoder_output, trg_mask)

    @torch.no_grad()
    def inference(self, input, trg):
        encoder_output = self.encoder(input)
        batch_size, num_tokens, max_len = input.shape[0], self.decoder.num_tokens, trg.shape[1] #(max_len = ML)
        output = torch.zeros(batch_size, max_len, num_tokens).to(self.device) #(B, ML, NT)
        current_seq = trg[:, 0].clone().unsqueeze(1) #(B, 1)
        for t in range(1, max_len):
            mask = self.create_mask(current_seq)
            logits = self.decoder(current_seq, encoder_output, mask) #(B, L_out, NT)
            next_token = torch.max(logits[:, -1], dim=-1)[1].unsqueeze(1) #(B, 1)
            current_seq = torch.cat([current_seq, next_token], dim=-1)
            output[:, t, :] = logits[:, -1]
        #output: (B, ML)
        return output