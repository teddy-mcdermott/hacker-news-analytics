def get_model_identity(args):
    if args.load_model:
        return args.load_model
    return f"bertopic_{args.type}_days{args.days}_minscore{args.min_score}_max{args.max_items}"
