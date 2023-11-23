python pretrain-sup.py --dataset arxiv --save_path encoder_arxiv_sup.pt --root data
python pretrain.py --dataset arxiv --save_path encoder_arxiv.pt --pretrain_path encoder_arxiv_sup.pt --lr 0.005 --root data
python finetune.py --epochs 5 --pretrain_path encoder_arxiv.pt --embedding_save_path embedding_arxiv.pt --root data/icdm2023_session1_test
python cluster.py --embedding_path embedding_arxiv.pt --k 15 --seed 666 --runs 5 --output submit.txt
