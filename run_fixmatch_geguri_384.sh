python -m torch.distributed.launch --nproc_per_node 2 train_fixmatch_geguri_memory_efficient.py --model swsl_resnext101_32x4d --batch-size 8 --lr 0.003 --expand-labels --amp --num-epochs 20 --warmup 2048 --threshold 0.95 --lambda-u 1 --mu 5 --msd --sched cosine --img-size 384 --use-ema --out result_fixmatch_384 --validation-fold 0
python -m torch.distributed.launch --nproc_per_node 2 train_fixmatch_geguri_memory_efficient.py --model swsl_resnext101_32x4d --batch-size 8 --lr 0.003 --expand-labels --amp --num-epochs 20 --warmup 2048 --threshold 0.95 --lambda-u 1 --mu 5 --msd --sched cosine --img-size 384 --use-ema --out result_fixmatch_384 --validation-fold 1
python -m torch.distributed.launch --nproc_per_node 2 train_fixmatch_geguri_memory_efficient.py --model swsl_resnext101_32x4d --batch-size 8 --lr 0.003 --expand-labels --amp --num-epochs 20 --warmup 2048 --threshold 0.95 --lambda-u 1 --mu 5 --msd --sched cosine --img-size 384 --use-ema --out result_fixmatch_384 --validation-fold 2
python -m torch.distributed.launch --nproc_per_node 2 train_fixmatch_geguri_memory_efficient.py --model swsl_resnext101_32x4d --batch-size 8 --lr 0.003 --expand-labels --amp --num-epochs 20 --warmup 2048 --threshold 0.95 --lambda-u 1 --mu 5 --msd --sched cosine --img-size 384 --use-ema --out result_fixmatch_384 --validation-fold 3
python -m torch.distributed.launch --nproc_per_node 2 train_fixmatch_geguri_memory_efficient.py --model swsl_resnext101_32x4d --batch-size 8 --lr 0.003 --expand-labels --amp --num-epochs 20 --warmup 2048 --threshold 0.95 --lambda-u 1 --mu 5 --msd --sched cosine --img-size 384 --use-ema --out result_fixmatch_384 --validation-fold 4
