import torch
import torch.nn.functional as F



def flash_attn(q , k , v , block_size = 64):
    #shape of q , k , v = batch_size , seq_len , head_dim
    #block_size = is our arbitrary block division to resemble the flash attention paper
    # this is the division of the sequence into smaller blocks to reduce memory usage



    B , T , D  = q.shape
    #B = batch_size
    #T = seq_len
    #D = head_dim

    output = torch.zeros_like(q) #a tensor to hold the output
    scale = 1.0 / (D ** 0.5) #scaling factor for the attention scores (underroot dimension)


    for start in range(0 , T , block_size):
        end = min(start + block_size , T)
        q_blk = q[: , start:end , :] #block of queries, slices the seq q of size block_size

        numerator = torch.zeros(B , end - start , D).to(q.device) #numerator for the attention scores
        denominator = torch.zeros(B , end - start).to(q.device)

        for k_start in range(0 , T , block_size):
            k_end = min(k_start + block_size , T)
            k_blk = k[: , k_start:k_end , :]
            v_blk = v[: , k_start:k_end , :]
            #block of keys and values, slices the seq k and v of size block_size

            attn_score = torch.matmul(q_blk , k_blk.transpose(-2 , -1)) * scale
            attn_max = attn_score.max(dim=-1, keepdim=True).values
            attn_score = attn_score - attn_max
            weight = torch.exp(attn_score)

            weighted_v = torch.matmul(weight , v_blk)
            numerator += weighted_v
            denominator += weight.sum(dim=-1)
        
        output[:, start:end, :] = numerator / denominator.unsqueeze(-1) #output is the numerator divided by the denominator

    return output
