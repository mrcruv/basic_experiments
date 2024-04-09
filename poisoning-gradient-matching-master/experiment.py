import pickle
import torch
import forest
from torchvision.datasets import CIFAR10
from utils import sample_poisons_ids_subgroups, split_clean_ids_by_class, sample_clean_ids_subgroups, split_poison_delta
from torch.utils.data import Subset

torch.backends.cudnn.benchmark = forest.consts.BENCHMARK
torch.multiprocessing.set_sharing_strategy(forest.consts.SHARING_STRATEGY)

# Parse input arguments
args = forest.options().parse_args()
# 100% reproducibility?
if args.deterministic:
    forest.utils.set_deterministic()

if __name__ == "__main__":
    with open('poisons/results.pickle', 'rb') as filehandle:
        poison_results = pickle.load(filehandle)
    # print(poison_results)

    setup = forest.utils.system_startup(args)

    trainset = CIFAR10(root=args.data_path, train=True, download=True)
    validset = CIFAR10(root=args.data_path, train=False, download=True)
    poison_ids = list(poison_results["poisons"].keys())
    poison_delta = poison_results["poison_delta"]
    poisonset = Subset(trainset, poison_ids)

    n_classes = len(trainset.classes)
    n_train_samples = len(trainset)
    k = 2

    # Sample k disjoint subgroups of poison indexes
    poison_ids_subgroups = sample_poisons_ids_subgroups(poison_ids, k)
    # Split clean samples indexes by target class
    clean_ids_by_class = split_clean_ids_by_class(trainset, poison_ids)
    # Sample k disjoint subgroups clean samples indexes s.t. the subgroups obtained are (almost) balanced
    clean_ids_subgroups = sample_clean_ids_subgroups(clean_ids_by_class, n_classes, k)
    # Split the poison delta with respect to the poison ids subgroups
    poison_deltas = split_poison_delta(poisonset, poison_delta, poison_ids_subgroups, k)

    poisoned_models_stats, clean_model_stats, EPIC_models_stats = [], [], []
    poisoned_models, clean_models, EPIC_models = ([forest.Victim(args, setup=setup) for _ in range(k)],
                                                  [forest.Victim(args, setup=setup) for _ in range(k)],
                                                  [forest.Victim(args, setup=setup) for _ in range(k)])
    batch_size = poisoned_models[0].defs.batch_size
    augmentations = poisoned_models[0].defs.augmentations

    kettles = [forest.Kettle(args, batch_size, augmentations, setup=dict(device=torch.device('cpu'), dtype=torch.float),
                             train_ids=clean_ids_subgroups[i], poison_ids=poison_ids_subgroups[i],
                             poison_results=poison_results) for i in range(k)]
