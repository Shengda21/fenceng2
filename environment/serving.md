# Model serving

The LLM arms used OpenAI-compatible chat-completion endpoints served by vLLM 0.28.0 on a single H800 80 GB GPU. The recorded model ids are `gpt-oss-20b`, `gpt-oss-120b`, and `qwen3.8-27b`.

The job scripts set `VLLM_USE_FLASHINFER_SAMPLER=0` for all models. The Qwen run also set `VLLM_USE_DEEP_GEMM=0`. Server command parameters were: max model length 8192, max number of sequences 64, GPU memory utilization 0.35 for `gpt-oss-20b`, 0.90 for `gpt-oss-120b`, and 0.55 for `qwen3.8-27b`. Hostnames, ports, usernames, and local paths have been omitted.

Request parameters recorded in the released calls include `temperature`, `max_tokens`, `extra_body`, token usage, call latency, and the vLLM `system_fingerprint`. The `gpt-oss` requests used `reasoning_effort: low` when present. The Qwen requests used `chat_template_kwargs: {enable_thinking: false}` when present.

## Sandbox reruns

The sandbox arms with one and five examples, and the zero-shot sandbox arms as a control, were also run
on one RTX PRO 6000 Blackwell 96 GB GPU (compute capability 12.0) with vLLM 0.28.0 (torch 2.13.0, CUDA 13.0), max model
length 8192, max number of sequences 128, and the Triton attention backend (`--attention-backend TRITON_ATTN`), which
this GPU needs because the default FlashInfer kernels are not built for it. `gpt-oss-120b` was served alone at GPU memory
utilization 0.92; `gpt-oss-20b` and `qwen3.8-27b` shared the GPU at 0.30 and 0.60. Request parameters were those of the
main runs. The commands are in `data/checks/commands/`, and the machine's software versions in
`data/checks/linux_rerun/env_info.txt`.

## Briefing line

The LLM arms of the briefing line were served with vLLM 0.28.0 on one RTX PRO 6000 Blackwell 96 GB GPU, with the Triton
attention backend, max model length 8192 and max number of sequences 128: `gpt-oss-120b` alone, and `gpt-oss-20b`
together with `qwen3.8-27b` at GPU memory utilization 0.30 and 0.60. The arms whose names end in `-long` were served
one model at a time with max model length 32,768. The serving logs, which list every non-default server argument, are
in `data/briefing/tasks/*/logs/`; every call record carries its own request parameters.
