#include <torch/extension.h>

void launch_flash_attn(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor output) {
    int seq_len = q.size(0);
    int head_dim = q.size(1);

    dim3 block(32); 
    dim3 grid((seq_len + block.x - 1) / block.x);

    flash_attn_kernel<<<grid, block>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        output.data_ptr<float>(),
        seq_len,
        head_dim
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn", &launch_flash_attn, "FlashAttention CUDA kernel");
}
