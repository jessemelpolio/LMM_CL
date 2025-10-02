OneVision sequential tuning scripts (TimeClock-first order)

Task order:
  TimeClock -> TextVQA -> PathVQA -> PixmoCount -> CUB200

Scripts:
- tune_full.sh            : Tune vision encoder + projector + LLM
- tune_vision_encoder.sh  : Tune vision encoder only
- tune_projector.sh       : Tune projector only
- tune_llm.sh             : Tune the full LLM
- tune_sa_proj.sh         : Tune only LLM self-attention proj (output proj)
- tune_sa_qkv.sh          : Tune only LLM self-attention QKV
- tune_mlp.sh             : Tune only LLM MLP blocks
- tune_mlp_gate_up.sh     : Tune only LLM MLP Gate & Up

Usage example:
  sbatch LLaVA-NeXT/slurms/final_experiments/onevision/tune_parts_tc_first/tune_full.slurm

Environment:
- Adapted from latest llama3 script conventions for OneVision evaluation.
- Set USE_TORCH_COMPILE=1 to enable torch.compile when Triton is present.
