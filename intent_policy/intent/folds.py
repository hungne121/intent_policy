"""Cross-fitting folds of the episodes (docs/requirements/INTENT_ACT_v3.2.md §4.2c): the measuring device is trained
5 times, each model on the episodes of 3 folds (validation: 1 fold, test: 1 fold), and every episode is measured by
the model that did not see it (`fold_of(seed)` = the model's test fold)."""
FOLDS = 5


def fold_of(seed: int, folds: int = FOLDS) -> int:
    """Fold of an episode: by its list cell (seeds are cell base + attempt, the bases are multiples of 10; the late and
    the early episode of a cell share the seed, hence the fold)."""
    return (int(seed) // 10) % int(folds)
