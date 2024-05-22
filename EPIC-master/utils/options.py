import argparse

def options():
    parser = argparse.ArgumentParser(description='PyTorch EPIC Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'tinyimagenet'],
                        help='dataset name')
    parser.add_argument('--arch', default='resnet18', type=str, choices=['resnet18', 'vgg16'],
                        help='dataset name')
    parser.add_argument('--epochs', default=200, type=int,
                        help='number of epochs', choices=(200, 40, 80, 20))
    parser.add_argument('--batch-size', default=128, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    parser.add_argument('--no_aug', action='store_true', help='do not use augmentation')
    parser.add_argument('--ap_model', action='store_true', help='use resnet in ap code')

    # Poison Setting
    parser.add_argument('--clean', action='store_true', help='train with the clean data')
    parser.add_argument("--poisons_path", type=str, help="where are the poisons?")
    parser.add_argument("--patch_size", type=int, default=5, help="Size of the patch")
    parser.add_argument('--trigger_path', type=str, default=None, help='path to the trigger')
    parser.add_argument("--backdoor", action='store_true', help='whether we are using backdoor attack')

    # Medoid Selection
    parser.add_argument('--greedy', default='LazyGreedy', choices=('LazyGreedy', 'StochasticGreedy'),
                        help='optimizer for subset selection')
    parser.add_argument('--metric', default='euclidean', choices=('euclidean', 'cosine'),
                        help='metric for subset selection')
    parser.add_argument('--subset_size', '-s', dest='subset_size', type=float, help='size of the subset', default=1.0)
    parser.add_argument('--subset_sampler', type=str, default='coreset-pred', choices=('random', 'coreset-pred'),
                        help='algorithm to select subsets')
    parser.add_argument('--subset_freq', type=int, default=10, help='frequency to update the subset')
    parser.add_argument('--equal_num', default=False, help='select equal numbers of examples from different classes')
    parser.add_argument('--scenario', default='scratch', choices=('scratch', 'transfer', 'finetune'),
                        help='select the training setting')
    parser.add_argument('--top_frac', type=float, default=0.1, help='fraction of low-confidence poisons to keep/drop')

    # Data Pruning
    parser.add_argument('--cluster_thresh', type=float, default=1., help='thrshold to drop examples in the subset')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--drop_after', type=int, default=1, help='dropping small clusters after this epoch')
    parser.add_argument('--stop_after', type=int, default=200, help='stop dropping small clusters after this epoch')
    parser.add_argument('--drop_mile', action='store_true', help='dropping small clusters at specific epoch')

    return parser