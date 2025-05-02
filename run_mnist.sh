python main.py --model vit_ape --data-set MNIST --data-path data --output_dir output/MNIST/vit_ape --batch-size 512 --epochs 25 --input-size 28 --lr 1e-4 --unscale-lr --repeated-aug

python main.py --model vit_ape_reg_rpe --data-set MNIST --data-path data --output_dir output/MNIST/vit_ape_reg_rpe --batch-size 512 --epochs 25 --input-size 28 --lr 1e-4 --unscale-lr --repeated-aug

python main.py --model vit_ape_poly_rpe --data-set MNIST --data-path data --output_dir output/MNIST/vit_ape_poly_rpe --batch-size 512 --epochs 25 --input-size 28 --lr 1e-4 --unscale-lr --repeated-aug

python main.py --model vit_ape_axial_rope --data-set MNIST --data-path data --output_dir output/MNIST/vit_ape_axial_rope --batch-size 512 --epochs 25 --input-size 28 --lr 1e-4 --unscale-lr --repeated-aug

python main.py --model vit_ape_mixed_rope --data-set MNIST --data-path data --output_dir output/MNIST/vit_ape_mixed_rope --batch-size 512 --epochs 25 --input-size 28 --lr 1e-4 --unscale-lr --repeated-aug
