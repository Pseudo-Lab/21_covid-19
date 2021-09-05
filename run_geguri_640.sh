python -m torch.distributed.launch --nproc_per_node 2 train_classifier_geguri.py --model swsl_resnext101_32x4d --amp --batch-size 16 --lr 0.00003 --num-epochs 40 --warmup 512 --msd --sched cosine --img-size 640 --out result_geguri_640 --validation-fold 0
python -m torch.distributed.launch --nproc_per_node 2 train_classifier_geguri.py --model swsl_resnext101_32x4d --amp --batch-size 16 --lr 0.00003 --num-epochs 40 --warmup 512 --msd --sched cosine --img-size 640 --out result_geguri_640 --validation-fold 1
python -m torch.distributed.launch --nproc_per_node 2 train_classifier_geguri.py --model swsl_resnext101_32x4d --amp --batch-size 16 --lr 0.00003 --num-epochs 40 --warmup 512 --msd --sched cosine --img-size 640 --out result_geguri_640 --validation-fold 2
python -m torch.distributed.launch --nproc_per_node 2 train_classifier_geguri.py --model swsl_resnext101_32x4d --amp --batch-size 16 --lr 0.00003 --num-epochs 40 --warmup 512 --msd --sched cosine --img-size 640 --out result_geguri_640 --validation-fold 3
python -m torch.distributed.launch --nproc_per_node 2 train_classifier_geguri.py --model swsl_resnext101_32x4d --amp --batch-size 16 --lr 0.00003 --num-epochs 40 --warmup 512 --msd --sched cosine --img-size 640 --out result_geguri_640 --validation-fold 4