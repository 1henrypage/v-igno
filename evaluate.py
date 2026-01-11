#!/usr/bin/env python3
"""
Evaluation script for inverse problems.

Runs IGNO-style gradient inversion (or encoder-based) on test set
and computes metrics.

All models (encoder, decoders, NF) are loaded from pretrained checkpoint.
No need to specify architecture in config.

Usage:
    python evaluate.py --config configs/evaluate.yaml
    python evaluate.py --config configs/evaluate.yaml --n-obs 25
    python evaluate.py --config configs/evaluate.yaml --snr-db 50
    python evaluate.py --config configs/evaluate.yaml --method encoder
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
from tqdm import trange

from src.solver.config import TrainingConfig
from src.problems import create_problem
from src.evaluation import IGNOInverter, EncoderInverter, compute_all_metrics, aggregate_metrics


def load_pretrained(checkpoint_path: Path, problem, device):
    """
    Load all pretrained models from checkpoint.

    The checkpoint contains:
    - models: dict of model state_dicts (enc, u, a, nf, etc.)
    - nf_config: NFConfig used to create the NF (optional, for reconstruction)

    Returns:
        model_dict: Dict of loaded models (excluding NF)
        nf: Loaded RealNVP model
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model state dicts
    if 'models' in checkpoint:
        ckpt_models = checkpoint['models']
    else:
        ckpt_models = checkpoint

    # Load NF - need to reconstruct from saved config or state dict
    if 'nf' not in ckpt_models:
        raise KeyError("Checkpoint does not contain 'nf' model")

    # Try to get NF config from checkpoint, otherwise infer from state dict
    nf_state = ckpt_models['nf']

    if 'nf_config' in checkpoint:
        # NF config was saved in checkpoint
        from src.solver.config import NFConfig
        nf_config = NFConfig(**checkpoint['nf_config'])
    else:
        # Infer config from state dict
        # The first layer weight shape tells us the dim
        first_key = [k for k in nf_state.keys() if 'st_net.net.0.weight' in k][0]
        dim = nf_state[first_key].shape[1] * 2  # cond_dim is dim//2

        # Count number of flows
        num_flows = len([k for k in nf_state.keys() if '.st_net.net.0.weight' in k])

        # Get hidden dim from first layer
        hidden_dim = nf_state[first_key].shape[0]

        # Count layers in st_net
        layer_keys = [k for k in nf_state.keys() if 'flows.0.st_net.net' in k and 'weight' in k]
        num_layers = len(layer_keys)

        from src.solver.config import NFConfig
        nf_config = NFConfig(
            dim=dim,
            num_flows=num_flows,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        print(f"  Inferred NF config: dim={dim}, flows={num_flows}, hidden={hidden_dim}, layers={num_layers}")

    # Create and load NF
    from src.components.nf import RealNVP
    nf = RealNVP(config=nf_config).to(device)
    nf.load_state_dict(nf_state)
    nf.eval()
    print(f"  Loaded: nf")

    # Load other models into problem.model_dict
    # We need to create the model architectures first
    # This is problem-specific, so we expect the problem to handle it

    # For now, we'll use a generic approach: save/load entire models
    # The checkpoint should contain the full model objects or we need the architecture

    # IMPORTANT: The problem must define how to create model architectures
    # For evaluation, we load state dicts into existing model structures

    loaded_models = {}
    for name, state_dict in ckpt_models.items():
        if name != 'nf':
            # Check if model exists in problem.model_dict
            if name in problem.model_dict:
                problem.model_dict[name].load_state_dict(state_dict)
                problem.model_dict[name].to(device)
                problem.model_dict[name].eval()
                loaded_models[name] = problem.model_dict[name]
                print(f"  Loaded: {name}")
            else:
                print(f"  Warning: {name} in checkpoint but not in problem.model_dict")

    return loaded_models, nf


def load_pretrained_full(checkpoint_path: Path, device):
    """
    Load all models directly from checkpoint (models saved as full objects).

    This is an alternative approach where the checkpoint contains
    the full model objects, not just state_dicts.

    Returns:
        model_dict: Dict of loaded models
        nf: Loaded RealNVP model
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    ckpt_models = checkpoint.get('models', checkpoint)

    # Separate NF from other models
    nf = None
    model_dict = {}

    for name, model_or_state in ckpt_models.items():
        if isinstance(model_or_state, nn.Module):
            # Full model was saved
            model_or_state.to(device)
            model_or_state.eval()
            if name == 'nf':
                nf = model_or_state
            else:
                model_dict[name] = model_or_state
            print(f"  Loaded (full model): {name}")
        elif isinstance(model_or_state, dict):
            # State dict was saved - need to reconstruct
            if name == 'nf':
                # Reconstruct NF
                nf_state = model_or_state

                # Infer config from state dict
                flow_keys = [k for k in nf_state.keys() if 'st_net.net.0.weight' in k]
                if flow_keys:
                    first_key = flow_keys[0]
                    dim = nf_state[first_key].shape[1] * 2
                    num_flows = len([k for k in nf_state.keys() if '.st_net.scale_layer.weight' in k])
                    hidden_dim = nf_state[first_key].shape[0]

                    from src.solver.config import NFConfig
                    from src.components.nf import RealNVP

                    # Count layers
                    layer_keys = [k for k in nf_state.keys() if 'flows.0.st_net.net' in k and 'weight' in k]
                    num_layers = len(layer_keys)

                    nf_config = NFConfig(dim=dim, num_flows=num_flows, hidden_dim=hidden_dim, num_layers=num_layers)
                    nf = RealNVP(config=nf_config).to(device)
                    nf.load_state_dict(nf_state)
                    nf.eval()
                    print(f"  Loaded (state dict): nf - dim={dim}, flows={num_flows}")

    return model_dict, nf


def save_results(results: dict, config: TrainingConfig, output_dir: Path):
    """Save results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    snr_str = f"snr{config.evaluation.snr_db}" if config.evaluation.snr_db else "clean"
    filename = f"eval_{config.evaluation.method}_nobs{config.evaluation.n_obs}_{snr_str}_{timestamp}.json"

    output_path = output_dir / filename

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    return output_path


def print_results(results: dict):
    """Pretty print evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for field in ['u', 'a']:
        if f'{field}_metrics' in results:
            print(f"\n{field.upper()} Metrics:")
            metrics = results[f'{field}_metrics']
            for metric_name, values in metrics.items():
                print(f"  {metric_name}:")
                print(f"    mean: {values['mean']:.6f}")
                print(f"    std:  {values['std']:.6f}")
                print(f"    min:  {values['min']:.6f}")
                print(f"    max:  {values['max']:.6f}")


def evaluate(config: TrainingConfig, verbose: bool = True):
    """
    Run evaluation on test set.

    Args:
        config: Configuration with evaluation settings
        verbose: Print progress

    Returns:
        Results dictionary
    """
    eval_cfg = config.evaluation
    device = torch.device(config.device)

    # Get pretrained path
    ckpt_path = config.get_pretrained_path()
    if ckpt_path is None:
        raise ValueError("pretrained.path must be set in config for evaluation")

    # Create problem (loads test data only)
    print(f"Creating problem: {config.problem.type}")
    problem = create_problem(config, load_train_data=False)

    # Load all models from checkpoint
    model_dict, nf = load_pretrained_full(ckpt_path, device)

    # Set models in problem
    problem.set_models(model_dict)

    # Create inverter
    if eval_cfg.method == 'igno':
        print("\nUsing IGNO gradient-based inversion")
        inverter = IGNOInverter(problem, nf)
    elif eval_cfg.method == 'encoder':
        print("\nUsing encoder-based inversion")
        raise NotImplementedError("Encoder-based inversion not yet implemented. Train encoder first.")
    else:
        raise ValueError(f"Unknown evaluation method: {eval_cfg.method}")

    # Setup observation indices (same for all samples)
    n_points = problem.get_n_points()
    n_test = problem.get_n_test_samples()

    obs_indices = problem.sample_observation_indices(
        n_total=n_points,
        n_obs=eval_cfg.n_obs,
        method=eval_cfg.obs_sampling,
        seed=eval_cfg.obs_seed,
    )

    print(f"\nEvaluation setup:")
    print(f"  Test samples: {n_test}")
    print(f"  Grid points: {n_points}")
    print(f"  Observations: {eval_cfg.n_obs} ({eval_cfg.obs_sampling} sampling)")
    print(f"  SNR: {eval_cfg.snr_db} dB" if eval_cfg.snr_db else "  SNR: Clean (no noise)")
    print(f"  Inversion epochs: {eval_cfg.inversion.epochs}")
    print(f"  Loss weights: PDE={eval_cfg.inversion.loss_weights.pde}, Data={eval_cfg.inversion.loss_weights.data}")

    # Run inversion on each test sample
    all_metrics_u = []
    all_metrics_a = []
    per_sample_results = []

    print(f"\nRunning inversion on {n_test} samples...")

    for i in trange(n_test, desc="Evaluating", disable=not verbose):
        # Prepare observations for this sample
        obs_data = problem.prepare_observations(
            sample_idx=i,
            obs_indices=obs_indices,
            snr_db=eval_cfg.snr_db,
        )

        # Run inversion
        if eval_cfg.method == 'igno':
            beta_opt = inverter.invert(
                x_obs=obs_data['x_obs'],
                u_obs=obs_data['u_obs'],
                x_full=obs_data['x_full'],
                config=eval_cfg.inversion,
                verbose=False,
            )
        else:
            beta_opt = inverter.invert(
                x_obs=obs_data['x_obs'],
                u_obs=obs_data['u_obs'],
                x_full=obs_data['x_full'],
            )

        # Predict on full grid
        preds = problem.predict_from_beta(beta_opt, obs_data['x_full'])

        # Compute metrics
        metrics_u = compute_all_metrics(preds['u_pred'], obs_data['u_true'])
        metrics_a = compute_all_metrics(preds['a_pred'], obs_data['a_true'])

        all_metrics_u.append(metrics_u)
        all_metrics_a.append(metrics_a)

        per_sample_results.append({
            'sample_idx': i,
            'u_metrics': metrics_u,
            'a_metrics': metrics_a,
        })

    # Aggregate
    aggregated_u = aggregate_metrics(all_metrics_u)
    aggregated_a = aggregate_metrics(all_metrics_a)

    results = {
        'config': {
            'method': eval_cfg.method,
            'n_obs': eval_cfg.n_obs,
            'obs_sampling': eval_cfg.obs_sampling,
            'snr_db': eval_cfg.snr_db,
            'inversion_epochs': eval_cfg.inversion.epochs,
            'loss_weights': {
                'pde': eval_cfg.inversion.loss_weights.pde,
                'data': eval_cfg.inversion.loss_weights.data,
            },
        },
        'n_samples': n_test,
        'u_metrics': aggregated_u,
        'a_metrics': aggregated_a,
        'per_sample': per_sample_results,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate inverse problem')
    parser.add_argument('--config', required=True, help='Path to config YAML')
    parser.add_argument('--device', type=str, help='Override device')
    parser.add_argument('--pretrained', type=str, help='Override pretrained path')
    parser.add_argument('--method', choices=['igno', 'encoder'], help='Inversion method')
    parser.add_argument('--n-obs', type=int, help='Number of observations')
    parser.add_argument('--snr-db', type=float, help='SNR in dB (use 0 for clean)')
    parser.add_argument('--inversion-epochs', type=int, help='Inversion epochs')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--quiet', action='store_true', help='Less verbose output')

    args = parser.parse_args()

    # Load config
    config = TrainingConfig.load(args.config)

    # Apply CLI overrides
    if args.device:
        config.device = args.device
    if args.pretrained:
        config.pretrained = {'path': args.pretrained}
    if args.method:
        config.evaluation.method = args.method
    if args.n_obs:
        config.evaluation.n_obs = args.n_obs
    if args.snr_db is not None:
        config.evaluation.snr_db = args.snr_db if args.snr_db > 0 else None
    if args.inversion_epochs:
        config.evaluation.inversion.epochs = args.inversion_epochs
    if args.seed:
        config.seed = args.seed
        config.evaluation.obs_seed = args.seed

    print(f"Config: {args.config}")
    print(f"Device: {config.device}")
    print(f"Problem: {config.problem.type}")

    # Run evaluation
    results = evaluate(config, verbose=not args.quiet)

    # Print results
    print_results(results)

    # Save results
    output_dir = Path(config.evaluation.results_dir)
    save_results(results, config, output_dir)


if __name__ == '__main__':
    main()