import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="NovelCraft", choices=["NovelCraft"])
    # model hyperparameters
    parser.add_argument("--e_mag", type=float, default=16, help="Embedding magnitued")
    # training hyperparameters
    parser.add_argument("--num_epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr_e", type=float, default=1e-3,
                        help="Learning rate for embedding v(x)")
    parser.add_argument("--lr_c", type=float, default=1e-1,
                        help="Learning rate for linear classifier {w_y, b_y}")
    parser.add_argument("--lr_s", type=float, default=1e-1,
                        help="Learning rate for sigma")
    parser.add_argument("--lr_d", type=float, default=1e-3,
                        help="Learning rate for delta_j")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr_milestones", default=[25, 28, 30])
    # loss hyperparameters
    parser.add_argument("--w_ccg", type=float, default=2e-1,
                        help="CCG loss weight, lambda in Eq. (23)")
    parser.add_argument("--w_nll", type=float, default=1 / 4096,
                        help="Negative log-likelihood weight, gamma in Eq. (22)")
    args = parser.parse_args()
    # add dataset related args
    if args.dataset == "NovelCraft":
        args.num_classes = 5
    return args


def train_ndcc(args):
    # init dataset
    # init model
    # init optimizer
    # init tensorboard
    # training epochs
    pass


if __name__ == "__main__":
    args = get_args()
    train_ndcc(args)
