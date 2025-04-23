python main.py --model vit_ape --data-set CIFAR10 --data-path data --output_dir output/CIFAR10/vit_ape --batch-size 512 --epochs 400 --input-size 32 --lr 1e-4 --unscale-lr --repeated-aug

python main.py --model vit_ape_reg_rpe --data-set CIFAR10 --data-path data --output_dir output/CIFAR10/vit_ape_reg_rpe --batch-size 512 --epochs 400 --input-size 32 --lr 1e-4 --unscale-lr --repeated-aug

python main.py --model vit_ape_poly_rpe --data-set CIFAR10 --data-path data --output_dir output/CIFAR10/vit_ape_poly_rpe --batch-size 512 --epochs 400 --input-size 32 --lr 1e-4 --unscale-lr --repeated-aug

python main.py --model vit_ape_axial_rope --data-set CIFAR10 --data-path data --output_dir output/CIFAR10/vit_ape_axial_rope --batch-size 512 --epochs 400 --input-size 32 --lr 1e-4 --unscale-lr --repeated-aug

python main.py --model vit_ape_mixed_rope --data-set CIFAR10 --data-path data --output_dir output/CIFAR10/vit_ape_mixed_rope --batch-size 512 --epochs 400 --input-size 32 --lr 1e-4 --unscale-lr --repeated-aug
