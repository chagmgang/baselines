CUDA_VISIBLE_DEVICES=0 python3 main.py --job_name learner --task 0 &

CUDA_VISIBLE_DEVICES=-1 python3 main.py --job_name actor --task 0 &
CUDA_VISIBLE_DEVICES=-1 python3 main.py --job_name actor --task 1 &
CUDA_VISIBLE_DEVICES=-1 python3 main.py --job_name actor --task 2 &
CUDA_VISIBLE_DEVICES=-1 python3 main.py --job_name actor --task 3 &
CUDA_VISIBLE_DEVICES=-1 python3 main.py --job_name actor --task 4 &
CUDA_VISIBLE_DEVICES=-1 python3 main.py --job_name actor --task 5 &
CUDA_VISIBLE_DEVICES=-1 python3 main.py --job_name actor --task 6 &
CUDA_VISIBLE_DEVICES=-1 python3 main.py --job_name actor --task 7 &
