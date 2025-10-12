import monai


def determinism(args):
    seed = getattr(args, 'seed', None)

    if seed is None or 'None':
        return
    else:
        monai.utils.misc.set_determinism(seed=seed, use_deterministic_algorithms=True)

    return


