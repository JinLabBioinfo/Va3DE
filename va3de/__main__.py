from rich.console import Console
import numpy as np
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import shutil
import time
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from va3de.sc_args import parse_args
from va3de import version


console = Console()


def version_callback(print_version: bool) -> None:
    """Print the version of the package."""
    if print_version:
        console.print(f"[yellow]va3de[/] version: [bold blue]{version}[/]")


def app():
    parser = argparse.ArgumentParser()
    args, x, y, depths, batches, dataset, valid_dataset = parse_args(
        parser)
        
    from va3de.experiments.vade_experiment import VaDEExperiment

    if args.read_distribution is not None:
        mitotic_frac = dataset.reference.sort_index()['mitotic_frac']
        local_frac = dataset.reference.sort_index()['local_frac']
        features = {'batch': batches, 'depth': depths,
                    'mitotic': mitotic_frac, 'local': local_frac}
    else:
        features = {'batch': batches, 'depth': depths}
    if args.valid_clusters is not None:
        valid_mask = np.int32(
            dataset.reference['cluster'].isin(args.valid_clusters))
        features['valid'] = valid_mask
    for feat in dataset.reference.columns:
        if feat not in features and feat in ['age', 'sex', 'donor', 'region', 'subtype', 'cellclass']:
            features[feat] = dataset.reference.sort_index()[feat].values

    load_results = args.load_results

    if args.simulate and not args.append_simulated:
        console.print(f"[green]Embedding simulated data...[/]")
        console.print(
            f"[green]We need to downsample the bulk data, this only needs to be done once...[/]")
        os.makedirs('scools', exist_ok=True)
        new_name = f"{dataset.dataset_name}_{dataset.res_name}_{args.simulate_n}_{args.simulate_depth}"
        dataset.dataset_name = new_name
        for i in range(int(args.n_runs)):  # unique dataset for each experiment
            scool_file = f"{new_name}_rep{i}.scool"
            if scool_file in os.listdir('scools'):
                print(scool_file, 'already exists...')
                scool_file = 'scools/' + scool_file
            else:
                scool_file = 'scools/' + scool_file
                dataset.write_scool(
                    scool_file, simulate=args.simulate, n_proc=8)
        # update after we have generated first scool
        dataset.update_from_scool(scool_file)

    console.print("[bold green]Embedding data using:[/]")
    start_time = time.time()

        
    if args.strata_offset is not None:
        if args.strata_offset != 0:
            exp_name += f'>{args.strata_offset}'

    wandb_config = None
    if args.wandb:
        wandb_config = {
            **args.__dict__, 'dataset': dataset.dataset_name}

    if args.load_va3de_from is None:
        from va3de.methods.vade import train_va3de
        experiment = VaDEExperiment('va3de', x, y, features, dataset, encoder=None, eval_inner=False, other_args=args)
        for run_i in range(int(args.n_runs)):
            train_va3de(features, dataset, experiment, run_i, args, preprocessing=None,
                    load_results=load_results, wandb_config=wandb_config)
    else:
        experiment = VaDEExperiment('va3de', x, y, features, dataset, encoder=None, eval_inner=True, other_args=args)
        experiment.run(load=False, outer_iter=0, start_time=start_time, log_wandb=args.wandb)
    if args.no_cache:
        try:
            shutil.rmtree(f'va3de/{dataset.dataset_name}')
        except FileNotFoundError:
            pass
        try:
            shutil.rmtree(f'va3de_models/{dataset.dataset_name}')
        except FileNotFoundError:
            pass

    console.print("[bright_green]Done running embedding methods...[/]")

    if 'help' not in sys.argv[1]:
        console.print("[bright_red]Unrecognized main argument...[/]")
    else:
        console.print("[bright_green]Welcome to Va3DE![/]")
        console.print("[yellow]\tva3de --help\t| to get a full list of arguments[/]")



if __name__ == "__main__":
    app()
