#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32
#define HEAD_DIM 64

__global__ void flash_attn_kernel_no_tiling(
    const float* __restrict__ query,   // [seq_len, head_dim]
    const float* __restrict__ key,     // [seq_len, head_dim]
    const float* __restrict__ value,   // [seq_len, head_dim]
    float* __restrict__ output,        // [seq_len, head_dim]
    int seq_len,
    int head_dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 

    if (i >= seq_len) return;


    float q_i[HEAD_DIM];
    for (int d = 0; d < head_dim; ++d) {
        q_i[d] = query[i * head_dim + d];
    }

    float max_score = -1e9f;
    float scores[1024];  
    float exp_scores[1024];
    float sum_exp = 0.0f;


    for (int j = 0; j < seq_len; ++j) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q_i[d] * key[j * head_dim + d];
        }
        scores[j] = dot;
        if (dot > max_score) max_score = dot;
    }

    for (int j = 0; j < seq_len; ++j) {
        exp_scores[j] = expf(scores[j] - max_score);
        sum_exp += exp_scores[j];
    }

    float out[HEAD_DIM] = {0.0f};

    for (int j = 0; j < seq_len; ++j) {
        float weight = exp_scores[j] / sum_exp;
        for (int d = 0; d < head_dim; ++d) {
            out[d] += weight * value[j * head_dim + d];
        }
    }

    for (int d = 0; d < head_dim; ++d) {
        output[i * head_dim + d] = out[d];
    }
}




#define BLOCK_SIZE 32
#define HEAD_DIM 64
#define TILE_SIZE 32  // Tile size along sequence length

__global__ void flash_attn_tiled_kernel(
    const float* __restrict__ query,   // [seq_len, head_dim]
    const float* __restrict__ key,     // [seq_len, head_dim]
    const float* __restrict__ value,   // [seq_len, head_dim]
    float* __restrict__ output,        // [seq_len, head_dim]
    int seq_len,
    int head_dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= seq_len) return;


    float q_i[HEAD_DIM];
    for (int d = 0; d < head_dim; ++d) {
        q_i[d] = query[i * head_dim + d];
    }

    float out[HEAD_DIM] = {0.0f};

    float max_score = -1e9f;
    float scores_tile[TILE_SIZE];
    float exp_scores_tile[TILE_SIZE];
    float total_exp = 0.0f;

    __shared__ float tile_K[TILE_SIZE][HEAD_DIM];
    __shared__ float tile_V[TILE_SIZE][HEAD_DIM];

    for (int tile_start = 0; tile_start < seq_len; tile_start += TILE_SIZE) {
        int j = tile_start + threadIdx.y;

        if (j < seq_len && threadIdx.y < TILE_SIZE && threadIdx.x < head_dim) {
            tile_K[threadIdx.y][threadIdx.x] = key[j * head_dim + threadIdx.x];
            tile_V[threadIdx.y][threadIdx.x] = value[j * head_dim + threadIdx.x];
        }

        __syncthreads();

        for (int t = 0; t < TILE_SIZE; ++t) {
            int j_idx = tile_start + t;
            if (j_idx >= seq_len) break;

            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += q_i[d] * tile_K[t][d];
            }

            scores_tile[t] = dot;
            max_score = fmaxf(max_score, dot);
        }

        __syncthreads();
    }

    for (int tile_start = 0; tile_start < seq_len; tile_start += TILE_SIZE) {
        int j = tile_start + threadIdx.y;

        if (j < seq_len && threadIdx.y < TILE_SIZE && threadIdx.x < head_dim) {
            tile_K[threadIdx.y][threadIdx.x] = key[j * head_dim + threadIdx.x];
            tile_V[threadIdx.y][threadIdx.x] = value[j * head_dim + threadIdx.x];
        }

        __syncthreads();

        for (int t = 0; t < TILE_SIZE; ++t) {
            int j_idx = tile_start + t;
            if (j_idx >= seq_len) break;

            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += q_i[d] * tile_K[t][d];
            }

            float exp_score = expf(dot - max_score);
            total_exp += exp_score;

            for (int d = 0; d < head_dim; ++d) {
                out[d] += exp_score * tile_V[t][d];
            }
        }

        __syncthreads();
    }
    for (int d = 0; d < head_dim; ++d) {
        output[i * head_dim + d] = out[d] / total_exp;
    }
}

