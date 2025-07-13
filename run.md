```bash

python ./src/main.py --dataset single_mnist-8 --epochs 300 --batch_size 256 --lr 0.001

python ./src/main.py --dataset mnist --epochs 150 --batch_size 1024 --lr 0.005

python ./src/main.py --dataset celeba --epochs 500 --batch_size 1024 --lr 0.001

# -----------
python ./src/main.py --dataset single_mnist-8 --sample

python ./src/main.py --dataset celeba --sample

python ./src/utils/data_loader.py
```