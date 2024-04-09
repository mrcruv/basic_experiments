import copy
import math
import random

import torch


def sample_poisons_ids_subgroups(poison_ids, n_subgroups):
    _poison_ids = copy.deepcopy(poison_ids)
    n_poisons = len(_poison_ids)
    poison_ids_subgroups = []
    subgroup_min_size = math.floor(n_poisons / n_subgroups)
    remainder = n_poisons % n_subgroups

    for i in range(n_subgroups):
        current_poison_ids_subgroup = random.sample(_poison_ids, subgroup_min_size) if (i < (n_subgroups - remainder)) \
            else random.sample(poison_ids, subgroup_min_size + 1)
        _poison_ids = [id for id in _poison_ids if id not in current_poison_ids_subgroup]
        poison_ids_subgroups.append(current_poison_ids_subgroup)

    return poison_ids_subgroups


def split_clean_ids_by_class(trainset, poison_ids):
    n_train_samples = len(trainset)
    n_classes = len(trainset.classes)
    clean_ids = list(range(1, n_train_samples + 1))
    clean_ids = [id for id in clean_ids if id not in poison_ids]
    n_clean_samples = len(clean_ids)
    clean_ids_by_class = [[] for _ in range(n_classes)]

    for id in clean_ids:
        clean_ids_by_class[trainset.targets[id - 1]].append(id)

    return clean_ids_by_class


def sample_clean_ids_subgroups(clean_ids_by_class, n_classes, n_subgroups):
    _clean_ids_by_class = copy.deepcopy(clean_ids_by_class)
    n_clean_samples_by_class = [len(_clean_ids_by_class[i]) for i in range(n_classes)]
    clean_ids_subgroups = [[] for _ in range(n_subgroups)]
    sizes = []
    remainders = []

    for i in range(n_classes):
        sizes.append(math.floor(n_clean_samples_by_class[i] / n_subgroups))
        remainders.append(n_clean_samples_by_class[i] % n_subgroups)

    for i in range(n_classes):
        for j in range(n_subgroups):
            current_clean_ids_subgroup_from_current_class = random.sample(_clean_ids_by_class[i], sizes[i]) if j < (
                    n_subgroups - remainders[i]) \
                else random.sample(_clean_ids_by_class[i], sizes[i] + 1)
            clean_ids_subgroups[j].extend(current_clean_ids_subgroup_from_current_class)
            _clean_ids_by_class[i] = [id for id in _clean_ids_by_class[i] if
                                      id not in current_clean_ids_subgroup_from_current_class]

    return clean_ids_subgroups


def split_poison_delta(poisonset, poison_delta, poison_ids_subgroups, n_subgroups):
    poison_deltas_by_index = dict(zip(poisonset.indices, poison_delta))
    poison_deltas = []
    for i in range(n_subgroups):
        current_subgroup_poison_ids = []
        for id in poison_ids_subgroups[i]:
            current_subgroup_poison_ids.append(poison_deltas_by_index[id])
        poison_deltas.append(current_subgroup_poison_ids)
        poison_deltas[i] = torch.stack(poison_deltas[i])

    return poison_deltas
