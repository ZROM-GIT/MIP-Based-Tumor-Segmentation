# utils/import_utils.py

import importlib

def dynamic_import_from_path(path: str):
    """
    Dynamically imports a class from a full module path string.
    Example input: "trainers.segmentation.unetr_trainer.UNETRTrainer"
    """
    try:
        module_path, class_name = path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import '{path}': {e}")


def get_trainer_and_tester(args):
    """
    Dynamically imports Trainer and Tester classes from paths in args.

    Returns:
        TrainerClass or None
        TesterClass or None

    If args.train is False, returns None instead of TrainerClass.
    If args.test is False, returns None instead of TesterClass.

    Assumes:
        args.trainer: "trainers.segmentation.unetr_trainer.UNETRTrainer"
        args.tester:  "testers.segmentation.unetr_tester.UNETRTester"
    """
    if not hasattr(args, "trainer") or not hasattr(args, "tester"):
        raise ValueError("Arguments must include 'trainer' and 'tester' fields.")

    TrainerClass = dynamic_import_from_path(args.trainer) if args.train else None
    TesterClass = dynamic_import_from_path(args.tester) if args.test else None

    return TrainerClass, TesterClass
