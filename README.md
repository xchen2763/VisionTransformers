# VisionTransformers
Repository for Columbia University course projects IEOR 6617 and IEOR 4540

# Report in Overleaf
[Report](https://www.overleaf.com/project/68093a8cf2cf98a61326477e)

# IEOR 4540 Project Description
<img width="609" alt="image" src="https://github.com/user-attachments/assets/50d5cdcc-8a60-41f1-a362-908250fc1390" />

# Rotary Positional Encodings for ViT and Performer -- code

This folder is RoPE-ViT-Performer training code based on [RoPE](https://github.com/naver-ai/rope-vit)
and [Performer](https://github.com/google-research/google-research/tree/master/performer) codebase.

Regular ViT with absolute positional embedding (APE) is implemented in `models_v2.py`.

- Baseline models with RPE variants
  - `vit_ape`
  - `vit_ape_reg_rpe`
  - `vit_ape_poly_rpe`
  - `vit_ape_axial_rope`
  - `vit_ape_mixed_rope`

RoPE ViT is implemented in `models_v2_rope.py`.

- Models
  - `rope_axial_deit_small_patch8_LS`
  - `rope_axial_deit_base_patch8_LS`
  - `rope_mixed_deit_small_patch8_LS`
  - `rope_mixed_deit_base_patch8_LS`
  - `rope_axial_ape_deit_small_patch8_LS`
  - `rope_axial_ape_deit_base_patch8_LS`
  - `rope_mixed_ape_deit_small_patch8_LS`
  - `rope_mixed_ape_deit_base_patch8_LS`

ViT with APE and Performer mechanism is implemented in `models_v2_performer.py`.

- Models
  - `performer_small_patch8_LS`
  - `performer_base_patch8_LS`

ViT with RoPE and Performer mechanism is implemented in `models_v2_performer_rope.py`

- Models
  - `performer_rope_axial_small_LS`
  - `performer_rope_axial_base_LS`
  - `performer_rope_mixed_small_LS`
  - `performer_rope_mixed_base_LS`
  - `performer_rope_axial_ape_small_LS`
  - `performer_rope_axial_ape_base_LS`
  - `performer_rope_mixed_ape_small_LS`
  - `performer_rope_mixed_ape_base_LS`

## Reference
This project uses code from `rope-vit`, licensed under the Apache License 2.0.
See https://github.com/naver-ai/rope-vit.