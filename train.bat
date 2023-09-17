pipenv run torchrun --nproc-per-node=1 train.py --data-path "../ORS13402_no_audio" --model DiT-L --num-workers 1 --epochs 100 --global-batch-size 32 --ckpt-every 20000 --seq-len 128 --dist gloo
