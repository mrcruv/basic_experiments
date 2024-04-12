#!/usr/bin/env python3
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
    poison_ids = [idx.item() for idx in poison_results["poisons"].keys()]
    poison_delta = poison_results["poison_delta"]
    poisonset = Subset(trainset, poison_ids)

    n_classes = len(trainset.classes)
    n_train_samples = len(trainset)
    k = 2

    # Sample k disjoint subgroups of poison indexes
    print(f'Sampling {k} poison indexes subgroups')
    poison_ids_subgroups = sample_poisons_ids_subgroups(poison_ids, k)
    print('Done')
    # Split clean samples indexes by target class
    print(f'Sampling clean samples indexes by target class')
    clean_ids_by_class = split_clean_ids_by_class(trainset, poison_ids)
    print('Done')
    # Sample k disjoint subgroups clean samples indexes s.t. the subgroups obtained are (almost) balanced
    print(f'Sampling {k} clean samples indexes (balanced) subgroups')
    clean_ids_subgroups = sample_clean_ids_subgroups(clean_ids_by_class, n_classes, k)
    print('Done')
    # Split the poison delta with respect to the poison ids subgroups
    print(f'Splitting poison delta')
    poison_deltas = split_poison_delta(poisonset, poison_delta, poison_ids_subgroups, k)
    print('Done')

    poisoned_models_stats, clean_models_stats, EPIC_models_stats = [], [], []
    poisoned_models, clean_models, EPIC_models = ([forest.Victim(args, setup=setup) for _ in range(k)],
                                                  [forest.Victim(args, setup=setup) for _ in range(k)],
                                                  [forest.Victim(args, setup=setup) for _ in range(k)])
    batch_size = poisoned_models[0].defs.batch_size
    augmentations = poisoned_models[0].defs.augmentations

    kettles = [forest.Kettle(args, batch_size, augmentations, setup=setup,
                             train_ids=clean_ids_subgroups[i], poison_ids=poison_ids_subgroups[i],
                             poison_results=poison_results) for i in range(k)]

    for i in range(k):
        poisoned_models[i].retrain(kettles[i], poison_deltas[i])
        poisoned_models_stats.append(poisoned_models[i].validate(kettles[i], poison_deltas[i]))
        print(poisoned_models_stats[i])

    for i in range(k):
        clean_models[i].train(kettles[i])
        clean_models_stats.append(clean_models[i].validate(kettles[i], poison_deltas[i]))
        print(clean_models_stats[i])

