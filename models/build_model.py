from . import mvit_s, mvit_s_two, mvit_b_two, mvit_b


MODEL_TABLE = {
    'mvit_s': mvit_s,
    'mvit_b': mvit_b,
    'mvit_s_two': mvit_s_two,
    'mvit_b_two': mvit_b_two,
}


def build_model(args, test_mode=False):
    """
    Args:
        args: all options defined in opts.py and num_classes
        test_mode:
    Returns:
        network model
        architecture name
    """
    model = MODEL_TABLE[args.backbone_net](**vars(args))
    # network_name = model.network_name if hasattr(model, 'network_name') else args.backbone_net
    # arch_name = "{dataset}-{modality}-{arch_name}".format(
    #     dataset=args.dataset, modality=args.modality, arch_name=network_name)
    # arch_name += "-f{}".format(args.groups)
    #
    # # add setting info only in training
    # if not test_mode:
    #     arch_name += "-{}{}-bs{}-e{}".format(args.lr_scheduler, "-syncbn" if args.sync_bn else "",
    #                                          args.batch_size, args.epochs)
    return model
