import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self , d_model: int , vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size , d_model)
    
    def forward(self , x ):
         return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncoding:
    def __init__(self, d_model: int , seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)


        # Create a matrix of shape (seq_len , d_model)
        pe = torch.zeros(seq_len , d_model)
        # Create a vector of shape (seq_len , 1)
        position = torch.arange(0 , seq_len, dtype =  torch.float).unsqueeze(1)
        # tensor([
        # [0.],
        # [1.],
        # [2.],
        # [3.],
        # [4.]])

        div_term = torch.exp(torch.arange(0 , d_model , 2).float() * (-math.log(10000.0) / d_model))
        # tensor([0., 2., 4.]) take a vector which has even values * 1000 divide by d_model


        #Apply sine to even places
        pe[: , 0::2] = torch.sine(position * div_term)  
        pe[: , 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1 , seq_len , d_model)

        self.register_buffer('pe' , pe) # saves tthe tensor in the file
    
    def forward(self , x):
        x = x + (self.pe[: , :x.shape[1] , :]).requires_grad_(False)
        return self.dropout(x)
        
    
class LayerNormalization(nn.Module):
    def __init__(self , eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim = -1 , keepdim = True)
        std = x.std(dim = -1 , keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias



class FeedForwardBlock():
    def __init__(self , d_model: int , d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model , d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff , d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

        
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self , d_model: int , h: int , dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.h = h
        assert d_model % h == 0 , "d_model should be divisible by number of heads = d_k"
        self.d_k = d_model//h

        self.w_q = nn.Linear(d_model , d_model)
        self.w_k = nn.Linear(d_model , d_model)
        self.w_v = nn.Linear(d_model , d_model)

        self.w_o = nn.Linear(d_model , d_model)

    @staticmethod
    def attention(query , key , value , mask , dropout):
        d_k = query.shape[-1]
        #@ = matrix multiplication
        #transpose(-2 , -1) transpose the last 2 dimesnsions
        attention_scores = (query @ key.transpose(-2 , -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask == 0 , -1e9)
        attention_scores = attention_scores.softmax(dim = -1)   #(Batch , h , seq_len , seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)
        #(Batch , h , seq_len , d_k) -> (Batch , h , seq_len , seq_len)
        return (attention_scores @ value), attention_scores
        #Final size = (Batch , h , seq_len , d_k)


    #mask = if you want some words to not interact with other words you mask them
    def forward(self , q , k , v , mask):
        query = self.w_q(q)   # basically multiplying k , q , v by weight matrix. Final size =  (Batch , seq_len , d_model) -> (Batch , seq_len , d_model) 
        key = self.w_k(k)       #Final size =  (Batch , seq_len , d_model) -> (Batch , seq_len , d_model) 
        value = self.w_v(v)     #Final size =  (Batch , seq_len , d_model) -> (Batch , seq_len , d_model) 
        
        # [Batch , seq_len , d_model] -> [Batch , seq_len , h , d_k] -> (transpose step) (Batch , h , seq_len , d_k)
        query = query.view(query.shape[0] , query.shape[1] , self.h , self.d_k ).transpose(1 , 2)
        key = key.view(key.shape[0] , key.shape[1] , self.h , self.d_k ).transpose(1 , 2)
        value = value.view(value.shape[0] , value.shape[1] , self.h , self.d_k ).transpose(1 , 2)

        x , self.attention_scores = MultiHeadAttentionBlock.attention(query , key , value , mask , self.dropout)

        #(Batch , h , seq_len , d_k) -> (Batch , seq_len , h , d_k) -> (Batch , seq_len , d_model)
        x = x.transpose(1 , 2).contiguous().view(x.shape[0] , -1 , self.h * self.d_k)

        #(Batch , seq_len , d_model) -> (Batch , seq_len , d_model)
        return self.w_o(x)
    
class ResidualConnection(nn.Module):

    def __init__(self , dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self , x , sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self , dropout: float , self_attention_block : MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block

        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        # First residual connection with self attention
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        
        # Second residual connection with feed forward
        x = self.residual_connections[1](x, self.feed_forward_block)
        
        return x


#Since there are many encoder blocks this is encoder objects
class Encoder(nn.Module):
    def __init__(self, layers : nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self , x , mask):
        for layer in self.layers:
            x = layer(x , mask)
        return self.norm(x)




class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock , cross_attention_block: MultiHeadAttentionBlock , feed_forward_block: FeedForwardBlock, dropout : float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.Module(ResidualConnection(dropout) for _ in range(3))

    # src_mask = mask for encoder , tgt_mask = mask for decoder
    def forward(self , x , encoder_output , src_mask , tgt_mask):
        x = self.residual_connection[0](x , lambda x: self.self_attention_block(x , x , x , tgt_mask))
        x = self.residual_connection[1](x , lambda x : self.cross_attention_block(x , encoder_output , encoder_output , src_mask))
        x = self.residual_connection[2](x , self.feed_forward_block)
        return x
    

class Decoder(nn.Module):
    def __init__(self, layers : nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self , x , encoder_output , src_mask , tgt_mask ):
        for layer in self.layers:
            x = layer(x , encoder_output , src_mask , tgt_mask)
        return self.norm(x)
    

class ProjectionLayer(nn.Module):
    def __init__(self, d_model : int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model , vocab_size)
    
    def forward(self , x):
        # (Batch_size , seq_len , d_model) --> (Batch , seq_len , vocab_size)
        return torch.log_softmax(self.prof(x) , dim = -1)

#NOW WE HAVE ALL THE INGRIDIENTS OF THE TRANSFORMER




class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed:InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
