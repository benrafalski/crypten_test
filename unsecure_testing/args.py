from argparse import ArgumentParser, REMAINDER

def parse_args():
    """
    Helper function parsing the command line options
    """
    parser = ArgumentParser(
        description="PyTorch distributed training launch "
        "helper utilty that will spawn up "
        "parties for MPC scripts on AWS"
    )

    parser.add_argument(
        "-s",
        "--sepsis",
        action='store_true',
        help="Chose to use the sepsis dataset",
    )

    parser.add_argument(
        "-r",
        "--random",
        action='store_true',
        help="Chose to use the random dataset",
    )

    parser.add_argument(
        "-c",
        "--clients",
        type=int,
        required=True,
        help="Number of clients used for training",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=5,
        help="Number of rounds used for training",
    )

    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        default=10,
        help="Batch size used for training",
    )

    return parser.parse_args()