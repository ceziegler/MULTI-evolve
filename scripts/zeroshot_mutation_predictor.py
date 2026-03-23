#!/usr/bin/env python3
"""
Script to perform zero-shot mutation prediction using ESM models.

This script takes a wild-type FASTA sequence and performs zero-shot mutation
prediction using ESM protein language models. Optionally, if a PDB/CIF structure
file is provided, it also runs ESM-IF (inverse folding) predictions.

Example usage (sequence-only):
    python scripts/zeroshot_mutation_predictor.py \
        --wt-file proteins/my_protein/wt.fasta

Example usage (with structure):
    python scripts/zeroshot_mutation_predictor.py \
        --wt-file proteins/my_protein/wt.fasta \
        --pdb-file proteins/my_protein/structure.pdb \
        --chain-id A

Example with masked-marginals scoring:
    python scripts/zeroshot_mutation_predictor.py \
        --wt-file proteins/my_protein/wt.fasta \
        --scoring-strategy masked-marginals

Example excluding specific positions (supports ranges):
    python scripts/zeroshot_mutation_predictor.py \
        --wt-file proteins/my_protein/wt.fasta \
        --excluded-positions 1,14,41-50,112

Example generating double mutants from top 30 singles:
    python scripts/zeroshot_mutation_predictor.py \
        --wt-file proteins/my_protein/wt.fasta \
        --double 30
"""

import argparse
import os
import sys
from itertools import combinations

from Bio import SeqIO
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from multievolve import zero_shot_esm_dms, zero_shot_esm_if_dms


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Perform zero-shot mutation prediction using ESM models.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--wt-file',
        required=True,
        help='Path to the wild-type FASTA file'
    )
    parser.add_argument(
        '--pdb-file',
        required=False,
        default=None,
        help='Path to PDB/CIF structure file (enables ESM-IF predictions)'
    )
    parser.add_argument(
        '--chain-id',
        required=False,
        default='A',
        help='Chain ID in the PDB file (default: A)'
    )
    parser.add_argument(
        '--output-file',
        required=False,
        default=None,
        help='Output CSV file path (default: <wt-dir>/zeroshot_predictions.csv)'
    )
    parser.add_argument(
        '--scoring-strategy',
        required=False,
        default='wt-marginals',
        choices=['wt-marginals', 'masked-marginals'],
        help='Scoring strategy for ESM models (default: wt-marginals)'
    )
    parser.add_argument(
        '--excluded-positions',
        required=False,
        default=None,
        help='Positions to exclude from mutation. Accepts comma-separated values and/or ranges (e.g., "1,14,41-50,112")'
    )
    parser.add_argument(
        '--double',
        type=int,
        required=False,
        default=None,
        metavar='N',
        help='Generate double mutants from top N single mutants and run full ESM scoring on each'
    )

    args = parser.parse_args()

    # Process excluded positions (supports comma-separated and ranges)
    if args.excluded_positions:
        positions = set()
        for part in args.excluded_positions.split(','):
            part = part.strip()
            if '-' in part:
                start, end = part.split('-')
                positions.update(range(int(start), int(end) + 1))
            else:
                positions.add(int(part))
        args.excluded_positions = sorted(positions)
    else:
        args.excluded_positions = []

    return args


def add_derived_columns(df):
    """Add derived columns to the mutation DataFrame."""
    df['aa_mutation'] = df['mutations'].apply(lambda x: x[-1])
    df['aa_substitution_type'] = df['mutations'].apply(lambda x: f'{x[0]}-{x[-1]}')
    df['pos'] = df['mutations'].apply(lambda x: int(x[1:-1]))
    return df


def rename_esm_columns(df):
    """Rename ESM DataFrame columns with 'esm_' prefix for clarity."""
    rename_map = {}
    for col in df.columns:
        if col == 'mutations':
            continue
        elif col.startswith('model_'):
            # model_1_logratio -> esm_model_1_logratio
            rename_map[col] = f'esm_{col}'
        elif col == 'average_model_logratio':
            rename_map[col] = 'esm_average_logratio'
        elif col == 'total_model_pass':
            rename_map[col] = 'esm_total_pass'
    return df.rename(columns=rename_map)


def apply_mutations_to_sequence(wt_seq, mutations):
    """
    Apply one or more mutations to a sequence.

    Args:
        wt_seq: Wild-type sequence string
        mutations: List of mutation strings (e.g., ['A10G', 'K25R'])

    Returns:
        Mutated sequence string
    """
    seq_list = list(wt_seq)
    for mut in mutations:
        wt_aa = mut[0]
        pos = int(mut[1:-1]) - 1  # Convert to 0-indexed
        mt_aa = mut[-1]
        assert seq_list[pos] == wt_aa, f"Expected {wt_aa} at position {pos+1}, found {seq_list[pos]}"
        seq_list[pos] = mt_aa
    return ''.join(seq_list)


def generate_double_mutant_combinations(single_df, n_top):
    """
    Generate double mutant combinations from top N single mutants.

    Args:
        single_df: DataFrame with single mutant scores (must have 'mutations' and 'pos' columns)
        n_top: Number of top single mutants to combine

    Returns:
        List of tuples: (double_mut_id, single1, single2)
    """
    # Get top N single mutants by average log-ratio
    top_singles = single_df.nlargest(n_top, 'esm_average_logratio').copy()

    double_mutants = []
    for i, j in combinations(range(len(top_singles)), 2):
        row1 = top_singles.iloc[i]
        row2 = top_singles.iloc[j]

        pos1 = row1['pos']
        pos2 = row2['pos']

        # Skip if mutations are at the same position
        if pos1 == pos2:
            continue

        # Create double mutant identifier (sorted by position)
        if pos1 < pos2:
            mut_id = f"{row1['mutations']}_{row2['mutations']}"
            single1, single2 = row1['mutations'], row2['mutations']
        else:
            mut_id = f"{row2['mutations']}_{row1['mutations']}"
            single1, single2 = row2['mutations'], row1['mutations']

        double_mutants.append((mut_id, single1, single2))

    return double_mutants


def score_double_mutants_esm(wt_seq, double_mutants, scoring_strategy='wt-marginals',
                             model_locations=None):
    """
    Score double mutants using ESM models with full re-scoring.

    Args:
        wt_seq: Wild-type sequence
        double_mutants: List of (mut_id, single1, single2) tuples
        scoring_strategy: 'wt-marginals' or 'masked-marginals'
        model_locations: List of ESM model names

    Returns:
        DataFrame with double mutant ESM scores
    """
    from esm import pretrained

    if model_locations is None:
        model_locations = [
            'esm1v_t33_650M_UR90S_1',
            'esm1v_t33_650M_UR90S_2',
            'esm1v_t33_650M_UR90S_3',
            'esm1v_t33_650M_UR90S_4',
            'esm1v_t33_650M_UR90S_5',
            'esm2_t36_3B_UR50D'
        ]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # First, compute wild-type token probabilities for each model
    print("  Computing wild-type token probabilities...")
    wt_probs = []
    alphabets = []

    for model_location in tqdm(model_locations, desc="  Loading models"):
        model, alphabet = pretrained.load_model_and_alphabet(model_location)
        model.eval()
        model = model.to(device)
        alphabets.append(alphabet)

        batch_converter = alphabet.get_batch_converter()
        data = [('wt', wt_seq)]
        _, _, batch_tokens = batch_converter(data)

        with torch.no_grad():
            token_probs = torch.log_softmax(model(batch_tokens.to(device))['logits'], dim=-1)

        wt_probs.append(token_probs.cpu().numpy()[0])

    # Score each double mutant
    print(f"  Scoring {len(double_mutants)} double mutants...")
    results = []

    for mut_id, single1, single2 in tqdm(double_mutants, desc="  Scoring doubles"):
        # Create double mutant sequence
        double_seq = apply_mutations_to_sequence(wt_seq, [single1, single2])

        # Score with each model
        model_scores = []
        for idx, model_location in enumerate(model_locations):
            model, alphabet = pretrained.load_model_and_alphabet(model_location)
            model.eval()
            model = model.to(device)

            batch_converter = alphabet.get_batch_converter()
            data = [('double', double_seq)]
            _, _, batch_tokens = batch_converter(data)

            if scoring_strategy == 'wt-marginals':
                with torch.no_grad():
                    token_probs = torch.log_softmax(model(batch_tokens.to(device))['logits'], dim=-1)
                double_probs = token_probs.cpu().numpy()[0]
            else:  # masked-marginals
                all_token_probs = []
                for i in range(batch_tokens.size(1)):
                    batch_tokens_masked = batch_tokens.clone()
                    batch_tokens_masked[0, i] = alphabet.mask_idx
                    with torch.no_grad():
                        token_probs = torch.log_softmax(
                            model(batch_tokens_masked.to(device))['logits'], dim=-1
                        )
                    all_token_probs.append(token_probs[:, i])
                double_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0).cpu().numpy()[0]

            # Calculate log-ratio score for the double mutant
            # Sum of log-ratios at both mutation positions
            score = 0.0
            for single_mut in [single1, single2]:
                wt_aa = single_mut[0]
                pos = int(single_mut[1:-1]) - 1
                mt_aa = single_mut[-1]

                wt_encoded = alphabet.tok_to_idx[wt_aa]
                mt_encoded = alphabet.tok_to_idx[mt_aa]

                # Score = P(mutant|double_context) - P(wt|wt_context)
                score += double_probs[pos + 1, mt_encoded] - wt_probs[idx][pos + 1, wt_encoded]

            if not np.isfinite(score):
                score = 0.0
            model_scores.append(score)

        # Build result record
        record = {
            'mutations': mut_id,
            'single1': single1,
            'single2': single2,
        }
        for i, score in enumerate(model_scores):
            record[f'esm_model_{i+1}_logratio'] = score

        results.append(record)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Calculate average and pass/fail
    logratio_cols = [f'esm_model_{i+1}_logratio' for i in range(len(model_locations))]
    df['esm_average_logratio'] = df[logratio_cols].mean(axis=1)

    for i in range(len(model_locations)):
        df[f'esm_model_{i+1}_pass'] = (df[f'esm_model_{i+1}_logratio'] > 0).astype(int)

    pass_cols = [f'esm_model_{i+1}_pass' for i in range(len(model_locations))]
    df['esm_total_pass'] = df[pass_cols].sum(axis=1)

    # Sort by total pass then by average log-ratio
    df.sort_values(
        by=['esm_total_pass', 'esm_average_logratio'],
        ascending=[False, False],
        inplace=True
    )

    return df


def score_double_mutants_esmif(wt_seq, double_mutants, pdb_file, chain_id='A'):
    """
    Score double mutants using ESM-IF model.

    Args:
        wt_seq: Wild-type sequence
        double_mutants: List of (mut_id, single1, single2) tuples
        pdb_file: Path to PDB/CIF file
        chain_id: Chain ID in PDB file

    Returns:
        DataFrame with double mutant ESM-IF scores
    """
    import esm
    from esm import pretrained
    from esm.inverse_folding.util import CoordBatchConverter

    model, alphabet = pretrained.load_model_and_alphabet('esm_if1_gvp4_t16_142M_UR50')
    model = model.eval()

    structure = esm.inverse_folding.util.load_structure(pdb_file, chain_id)
    coords, native_seq = esm.inverse_folding.util.extract_coords_from_structure(structure)

    device = next(model.parameters()).device
    batch_converter = CoordBatchConverter(alphabet)

    # Get wild-type scores
    batch = [(coords, None, wt_seq)]
    coords_batch, confidence, strs, tokens, padding_mask = batch_converter(batch, device=device)
    prev_output_tokens = tokens[:, :-1].to(device)
    logits, _ = model.forward(coords_batch, padding_mask, confidence, prev_output_tokens)
    wt_scores = logits.detach().numpy()[0]

    # Score each double mutant
    results = []
    print(f"  Scoring {len(double_mutants)} double mutants with ESM-IF...")

    for mut_id, single1, single2 in tqdm(double_mutants, desc="  ESM-IF scoring"):
        double_seq = apply_mutations_to_sequence(wt_seq, [single1, single2])

        batch = [(coords, None, double_seq)]
        coords_batch, confidence, strs, tokens, padding_mask = batch_converter(batch, device=device)
        prev_output_tokens = tokens[:, :-1].to(device)
        logits, _ = model.forward(coords_batch, padding_mask, confidence, prev_output_tokens)
        double_scores = logits.detach().numpy()[0]

        # Calculate log-ratio
        score = 0.0
        for single_mut in [single1, single2]:
            wt_aa = single_mut[0]
            pos = int(single_mut[1:-1]) - 1
            mt_aa = single_mut[-1]

            wt_tok = alphabet.tok_to_idx[wt_aa]
            mt_tok = alphabet.tok_to_idx[mt_aa]

            score += double_scores[mt_tok, pos] - wt_scores[wt_tok, pos]

        if not np.isfinite(score):
            score = 0.0

        results.append({
            'mutations': mut_id,
            'esmif_logratio': score
        })

    return pd.DataFrame(results)


def main():
    args = parse_args()

    # Validate input file exists
    if not os.path.exists(args.wt_file):
        print(f"Error: Wild-type file not found: {args.wt_file}")
        sys.exit(1)

    # Read wild-type sequence
    wt_seq = str(SeqIO.read(args.wt_file, "fasta").seq)
    print(f"Read wild-type sequence: {len(wt_seq)} residues")

    # Determine output file path
    if args.output_file:
        output_file = args.output_file
    else:
        wt_dir = os.path.dirname(args.wt_file)
        if not wt_dir:
            wt_dir = '.'
        output_file = os.path.join(wt_dir, 'zeroshot_predictions.csv')

    # Run ESM zero-shot predictions for single mutants
    print("Running ESM zero-shot predictions for single mutants...")
    esm_df = zero_shot_esm_dms(wt_seq, scoring_strategy=args.scoring_strategy)
    esm_df = rename_esm_columns(esm_df)

    # Run ESM-IF predictions if PDB file provided
    if args.pdb_file:
        if not os.path.exists(args.pdb_file):
            print(f"Error: PDB file not found: {args.pdb_file}")
            sys.exit(1)

        print("Running ESM-IF zero-shot predictions for single mutants...")
        esmif_df = zero_shot_esm_if_dms(
            wt_seq,
            args.pdb_file,
            chain_id=args.chain_id,
            scoring_strategy=args.scoring_strategy
        )
        esmif_df = esmif_df.rename(columns={'logratio': 'esmif_logratio'})

        # Merge ESM and ESM-IF DataFrames
        result_df = pd.merge(esm_df, esmif_df[['mutations', 'esmif_logratio']],
                            on='mutations', how='outer')
    else:
        result_df = esm_df

    # Add derived columns
    result_df = add_derived_columns(result_df)

    # Filter out excluded positions
    if args.excluded_positions:
        before_count = len(result_df)
        result_df = result_df[~result_df['pos'].isin(args.excluded_positions)]
        after_count = len(result_df)
        print(f"Excluded {before_count - after_count} mutations at positions: {args.excluded_positions}")

    # Save single mutant results
    result_df.to_csv(output_file, index=False)
    print(f"\nSaved single mutant predictions to: {output_file}")

    # Generate and score double mutants if requested
    double_df = None
    if args.double:
        print(f"\n=== Double Mutant Analysis ===")
        print(f"Generating combinations from top {args.double} single mutants...")

        # Generate double mutant combinations
        double_mutants = generate_double_mutant_combinations(result_df, args.double)
        print(f"Generated {len(double_mutants)} double mutant combinations")

        if double_mutants:
            # Score with ESM models
            print("\nRunning ESM scoring on double mutants...")
            double_df = score_double_mutants_esm(
                wt_seq,
                double_mutants,
                scoring_strategy=args.scoring_strategy
            )

            # Score with ESM-IF if PDB provided
            if args.pdb_file:
                print("\nRunning ESM-IF scoring on double mutants...")
                esmif_double_df = score_double_mutants_esmif(
                    wt_seq,
                    double_mutants,
                    args.pdb_file,
                    chain_id=args.chain_id
                )
                double_df = pd.merge(
                    double_df,
                    esmif_double_df[['mutations', 'esmif_logratio']],
                    on='mutations',
                    how='left'
                )

            # Save double mutant results
            double_output = output_file.replace('.csv', '_doubles.csv')
            double_df.to_csv(double_output, index=False)
            print(f"\nSaved double mutant predictions to: {double_output}")

    # Print summary statistics
    print("\n=== Single Mutant Summary ===")
    print(f"Total single mutations analyzed: {len(result_df)}")

    # ESM statistics
    if 'esm_average_logratio' in result_df.columns:
        avg_score = result_df['esm_average_logratio'].mean()
        beneficial = (result_df['esm_average_logratio'] > 0).sum()
        print(f"\nESM predictions:")
        print(f"  Mean log-ratio: {avg_score:.4f}")
        print(f"  Beneficial mutations (score > 0): {beneficial}")

    if 'esm_total_pass' in result_df.columns:
        all_pass = (result_df['esm_total_pass'] == result_df['esm_total_pass'].max()).sum()
        print(f"  Mutations passing all models: {all_pass}")

    # ESM-IF statistics
    if 'esmif_logratio' in result_df.columns:
        avg_esmif = result_df['esmif_logratio'].mean()
        beneficial_esmif = (result_df['esmif_logratio'] > 0).sum()
        print(f"\nESM-IF predictions:")
        print(f"  Mean log-ratio: {avg_esmif:.4f}")
        print(f"  Beneficial mutations (score > 0): {beneficial_esmif}")

    # Top single mutations
    print(f"\nTop 10 single mutations by ESM average log-ratio:")
    top_cols = ['mutations', 'esm_average_logratio']
    if 'esmif_logratio' in result_df.columns:
        top_cols.append('esmif_logratio')
    if 'esm_total_pass' in result_df.columns:
        top_cols.append('esm_total_pass')

    top_df = result_df.nlargest(10, 'esm_average_logratio')[top_cols]
    print(top_df.to_string(index=False))

    # Double mutant statistics
    if double_df is not None and len(double_df) > 0:
        print(f"\n=== Double Mutant Summary ===")
        print(f"Total double mutants scored: {len(double_df)}")

        if 'esm_average_logratio' in double_df.columns:
            avg_double = double_df['esm_average_logratio'].mean()
            beneficial_double = (double_df['esm_average_logratio'] > 0).sum()
            print(f"\nESM predictions:")
            print(f"  Mean log-ratio: {avg_double:.4f}")
            print(f"  Beneficial double mutants (score > 0): {beneficial_double}")

        if 'esm_total_pass' in double_df.columns:
            max_pass = double_df['esm_total_pass'].max()
            all_pass_double = (double_df['esm_total_pass'] == max_pass).sum()
            print(f"  Double mutants passing all models: {all_pass_double}")

        if 'esmif_logratio' in double_df.columns:
            avg_esmif_double = double_df['esmif_logratio'].mean()
            beneficial_esmif_double = (double_df['esmif_logratio'] > 0).sum()
            print(f"\nESM-IF predictions:")
            print(f"  Mean log-ratio: {avg_esmif_double:.4f}")
            print(f"  Beneficial double mutants (score > 0): {beneficial_esmif_double}")

        print(f"\nTop 10 double mutants by ESM average log-ratio:")
        double_top_cols = ['mutations', 'esm_average_logratio']
        if 'esmif_logratio' in double_df.columns:
            double_top_cols.append('esmif_logratio')
        if 'esm_total_pass' in double_df.columns:
            double_top_cols.append('esm_total_pass')

        top_double_df = double_df.nlargest(10, 'esm_average_logratio')[double_top_cols]
        print(top_double_df.to_string(index=False))


if __name__ == '__main__':
    main()
