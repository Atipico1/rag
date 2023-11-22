from src.classes.prompt import TotalPromptSet


def evaluation_per_perturbation(model, dataset: TotalPromptSet, args):
    if args.perturbation_type == "context_swap":
        pass
    elif args.perturbation_type == "conflict":
        pass
    elif args.perturbation_type == "adversarial":
        pass
    elif args.perturbation_type == "entity_swap":
        pass
    else:
        raise NotImplementedError
