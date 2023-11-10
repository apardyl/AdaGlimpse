import argparse
import inspect

import lightning

import architectures
import datasets


def experiment_from_args(argv, prog_name='TrainWhereToLookNext',
                         add_argparse_args_fn=None, modify_args_fn=None):
    parser = argparse.ArgumentParser(
        prog=prog_name
    )

    archs = {k: v for k, v in inspect.getmembers(architectures, inspect.isclass) if
             issubclass(v, lightning.LightningModule) and not inspect.isabstract(v)}
    dss = {k: v for k, v in inspect.getmembers(datasets, inspect.isclass) if
           issubclass(v, lightning.LightningDataModule) and not inspect.isabstract(
               v) and k != 'LightningDataModule'}

    parser.add_argument('dataset', help='Dataset to use', choices=dss.keys())
    parser.add_argument('arch', help='Model architecture', choices=archs.keys())
    main_args = parser.parse_args(argv[1:3])

    arch_class = archs[main_args.arch]
    data_module_class = dss[main_args.dataset]

    parser = argparse.ArgumentParser(
        prog=f'{prog_name} {main_args.dataset} {main_args.arch}',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser = arch_class.add_argparse_args(parser)
    parser = data_module_class.add_argparse_args(parser)
    if add_argparse_args_fn is not None:
        parser = add_argparse_args_fn(parser)

    args = parser.parse_args(argv[3:])

    if modify_args_fn is not None:
        args = modify_args_fn(args)

    setattr(args, 'dataset', main_args.dataset)
    setattr(args, 'arch', main_args.arch)

    args_dict = vars(args)

    data_module = data_module_class(**args_dict)
    model = arch_class(data_module, **args_dict)

    return data_module, model, args
