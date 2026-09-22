# GPU Notebooks

`stockham_sweep_colab.ipynb` builds FFTop with CUDA on a Colab GPU and runs the
three-kernel sweep from `bench/config/gpu_stockham.yaml`:

- Cooley–Tukey (global memory)
- Stockham (global memory ping-pong)
- Stockham (shared-memory tiles / four-step for large N)

It plots runtime, effective bandwidth, estimated global-memory traffic, and
throughput (\(N / t\)).

Open the notebook from GitHub (`nguyen-trinhtk/FFTop`, `main`). The clone cell
pulls `https://github.com/nguyen-trinhtk/FFTop.git` into `/content/FFTop`.
