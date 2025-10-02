OneVision sequential tuning scripts (alternative task order)

Task order:
  PathVQA -> CUB200 -> TextVQA -> TimeClock -> PixmoCount

Scripts:
- tune_full.sh          : Tune vision encoder + projector + LLM
- tune_vision_encoder.sh: Tune vision encoder only
- tune_projector.sh     : Tune projector only
- tune_llm.sh           : Tune the full LLM
- tune_sa_proj.sh       : Tune only LLM self-attention projections
- tune_mlp.sh           : Tune only LLM MLP blocks

Usage example:
  bash tune_full.sh

Environment:
- Adapted from latest llama3 script conventions for OneVision.
- Set USE_TORCH_COMPILE=1 to enable torch.compile when Triton is present.
