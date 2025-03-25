# VisionTransformers
Repository for Columbia University course projects IEOR 6617 and IEOR 4540

# Rotary Positional Encodings for ViT and Performer -- code

This folder is RoPE-ViT-Performer training code based on [RoPE](https://github.com/naver-ai/rope-vit)
and [Performer](https://github.com/google-research/google-research/tree/master/performer) codebase.

Regular ViT with absolute positional embedding (APE) is implemented in `models_v2.py`.

- Models
  - `deit_small_patch8_LS`
  - `deit_base_patch8_LS`

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


### Training

```bash
python main.py --model deit_small_patch8_LS --data-set CIFAR --data-path ${data_path} --output_dir ${save_path} --batch-size 512 --epochs 400 --input-size 32 --lr 1e-4 --unscale-lr --repeated-aug
```