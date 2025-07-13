import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0", help="GPU id.")
    parser.add_argument("--dataset", type=str, help="Dataset name.")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate.")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--sample", action="store_true", help="Sample from the model.")
    args = parser.parse_args()

    return args
