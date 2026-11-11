# Third-party notices

The Apache 2.0 license in [`LICENSE`](LICENSE) covers the project's original code. The distribution includes these upstream components with their original licenses and notices.

| Component | Included material | License |
| --- | --- | --- |
| [SGLang at `eb73a50`](https://github.com/sgl-project/sglang/tree/eb73a50b1d6183deb1d79ca425677bd0cf93342f) | H3 inference patch. Installation fetches the pinned Python runtime. | [Apache 2.0](vendor/sglang.LICENSE) and additional notices alongside individual kernels |
| [SageAttention at `d1a57a5`](https://github.com/thu-ml/SageAttention/tree/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5) | SageAttention2 and optional SageAttention3 sources, with C++20 build changes | [Apache 2.0](vendor/SageAttention/LICENSE) |
| [NVIDIA CUTLASS at `b2f249a`](https://github.com/NVIDIA/cutlass/tree/b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd) | Headers for the optional SageAttention3 build | [BSD 3-Clause](vendor/SageAttention/sageattention3_blackwell/csrc/cutlass/LICENSE.txt) |

Locally modified upstream files carry a modification notice. Source revisions and implementation references are recorded in [`provenance/sources.json`](provenance/sources.json).

## Model files and results

The runtime can use separately obtained MiniMax-H3-Image weights and the MiniMax-H3 text encoder. Each model has a MiniMax Community License Agreement that applies to model use, distribution, and outputs. The agreements exclude the European Union, the United Kingdom, South Korea, and the United States from their territory grants and describe how to request authorization from MiniMax. [MiniMax-H3-Image agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE) · [MiniMax-H3 agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
