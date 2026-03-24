#!/usr/bin/env python3
"""
Script to perform zero-shot mutation prediction using ESM models.

This script takes a wild-type FASTA sequence and performs zero-shot mutation
prediction using ESM protein language models. Optionally, if a PDB/CIF structure
file is provided, it also runs ESM-IF (inverse folding) predictions.

Scoring Methods:
- Non structure-informed: Ensemble of 5 ESM-1v models + ESM2 3B (wt-marginals)
- Structure-informed: ESM-IF1 GVP-Transformer inverse-folding model

Score Types:
- Fold-change (FC): Ratio of mutant to wild-type likelihood
- Z-score: Normalized FC by amino acid substitution type or mutation target

Ranking Strategies:
- ESM FC: Two-tier ranking by model consensus (f(y') > 0), then mean FC
- ESM-IF FC: Direct ranking by FC score
- Z-scores: Direct ranking by z-score (both methods)

Sampling Approaches:
- Top n: Select top n mutations from ranked list
- Position-exclusive: Select top n mutations with one per position

Example usage (sequence-only):
    python scripts/zeroshot_mutation_predictor.py \
        --wt-file proteins/my_protein/wt.fasta

Example usage (with structure):
    python scripts/zeroshot_mutation_predictor.py \
        --wt-file proteins/my_protein/wt.fasta \
        --pdb-file proteins/my_protein/structure.pdb \
        --chain-id A

Example with sampling:
    python scripts/zeroshot_mutation_predictor.py \
        --wt-file proteins/my_protein/wt.fasta \
        --pdb-file proteins/my_protein/structure.pdb \
        --n-variants 24

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
    parser.add_argument(
        '--n-variants',
        type=int,
        required=False,
        default=None,
        metavar='N',
        help='Number of top variants to sample from each ranking method'
    )
    parser.add_argument(
        '--z-score-grouping',
        required=False,
        default='aa_substitution_type',
        choices=['aa_substitution_type', 'aa_mutation'],
        help='Grouping method for z-score calculation: aa_substitution_type (e.g., A->P) or aa_mutation (e.g., ->P) (default: aa_substitution_type)'
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


def calculate_z_scores(df, score_col, grouping_col, z_col_name):
    """
    Calculate z-scores for a score column grouped by a categorical column.

    Z-score formula: z = (y' - μ_S) / σ_S
    where S is the set of mutations with the same grouping value.

    Args:
        df: DataFrame with mutations
        score_col: Column containing scores to normalize (e.g., 'esm_average_logratio')
        grouping_col: Column to group by (e.g., 'aa_substitution_type' or 'aa_mutation')
        z_col_name: Name for the output z-score column

    Returns:
        DataFrame with z-score column added (groups with < 5 samples are excluded)
    """
    result_dfs = []

    for group_value in df[grouping_col].unique():
        subset = df[df[grouping_col] == group_value].copy()

        # Only include groups with sufficient samples for meaningful statistics
        if len(subset) >= 5:
            mean_score = subset[score_col].mean()
            std_score = subset[score_col].std()

            if std_score > 0:
                subset[z_col_name] = (subset[score_col] - mean_score) / std_score
            else:
                subset[z_col_name] = 0.0

            result_dfs.append(subset)

    if result_dfs:
        return pd.concat(result_dfs, ignore_index=True)
    else:
        return pd.DataFrame()


def rank_esm_fc(df):
    """
    Rank mutations using ESM fold-change with two-tier ranking.

    Ranking strategy for non structure-informed FC scores:
    1. Calculate f(y') = number of models where y' > 1 (log-ratio > 0)
    2. Mutations with f(y') > 0: rank by f(y') descending, then by mean y' descending
    3. Mutations with f(y') = 0: rank by mean y' descending

    Args:
        df: DataFrame with 'esm_total_pass' and 'esm_average_logratio' columns

    Returns:
        DataFrame sorted by ESM FC ranking
    """
    df = df.copy()

    # Split into two groups: f(y') > 0 and f(y') = 0
    positive_pass = df[df['esm_total_pass'] > 0].copy()
    zero_pass = df[df['esm_total_pass'] == 0].copy()

    # Rank positive_pass by esm_total_pass desc, then esm_average_logratio desc
    positive_pass = positive_pass.sort_values(
        by=['esm_total_pass', 'esm_average_logratio'],
        ascending=[False, False]
    )

    # Rank zero_pass by esm_average_logratio desc only
    zero_pass = zero_pass.sort_values(
        by='esm_average_logratio',
        ascending=False
    )

    # Concatenate: positive_pass first, then zero_pass
    ranked = pd.concat([positive_pass, zero_pass], ignore_index=True)
    ranked['esm_fc_rank'] = range(1, len(ranked) + 1)

    return ranked


def rank_esmif_fc(df):
    """
    Rank mutations using ESM-IF fold-change.

    Ranking strategy for structure-informed FC scores:
    Direct ranking by y' (log-ratio) score.

    Args:
        df: DataFrame with 'esmif_logratio' column

    Returns:
        DataFrame sorted by ESM-IF FC ranking
    """
    df = df.copy()
    df = df.sort_values(by='esmif_logratio', ascending=False)
    df['esmif_fc_rank'] = range(1, len(df) + 1)
    return df


def rank_by_zscore(df, z_col, rank_col_name):
    """
    Rank mutations by z-score.

    Args:
        df: DataFrame with z-score column
        z_col: Name of z-score column to rank by
        rank_col_name: Name for the output rank column

    Returns:
        DataFrame sorted by z-score ranking
    """
    df = df.copy()
    df = df.sort_values(by=z_col, ascending=False)
    df[rank_col_name] = range(1, len(df) + 1)
    return df


def sample_top_n(df, n, rank_col):
    """
    Sample top n mutations from ranked list.

    Sampling approach 1: M = {(i, x'_i) ∈ R : rank((i, x'_i)) ≤ n}

    Args:
        df: DataFrame sorted by ranking
        n: Number of mutations to sample
        rank_col: Column containing rank values

    Returns:
        DataFrame with top n mutations
    """
    return df[df[rank_col] <= n].copy()


def sample_position_exclusive(df, n, rank_col, excluded_positions=None):
    """
    Sample top n mutations with position-specific exclusivity.

    Sampling approach 2: Select top n mutations while enforcing that only
    one mutation per position is included.

    M = {(i, x'_i) ∈ R : |M| ≤ n ∧ ∄(i, y_i) ∈ R where rank((i, y_i)) > rank((i, x'_i))}

    Args:
        df: DataFrame sorted by ranking
        n: Number of mutations to sample
        rank_col: Column containing rank values
        excluded_positions: List of positions to exclude

    Returns:
        DataFrame with top n position-exclusive mutations
    """
    if excluded_positions is None:
        excluded_positions = []

    sampled = []
    used_positions = set(excluded_positions)

    # Ensure sorted by rank
    df_sorted = df.sort_values(by=rank_col)

    for _, row in df_sorted.iterrows():
        pos = row['pos']
        if pos not in used_positions:
            sampled.append(row)
            used_positions.add(pos)
        if len(sampled) >= n:
            break

    if sampled:
        return pd.DataFrame(sampled)
    else:
        return pd.DataFrame()


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

    Uses ESM FC two-tier ranking to select top singles:
    - First by esm_total_pass (consensus)
    - Then by esm_average_logratio (magnitude)

    Args:
        single_df: DataFrame with single mutant scores (must have 'mutations' and 'pos' columns)
        n_top: Number of top single mutants to combine

    Returns:
        List of tuples: (double_mut_id, single1, single2)
    """
    # Get top N single mutants using ESM FC ranking
    ranked_df = rank_esm_fc(single_df)
    top_singles = ranked_df.head(n_top).copy()

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

    # ===========================================
    # RANKING: Apply all ranking strategies
    # ===========================================
    print("\n=== Applying Ranking Strategies ===")

    # 1. ESM FC ranking (two-tier: consensus then magnitude)
    print("Ranking by ESM FC (two-tier)...")
    result_df = rank_esm_fc(result_df)

    # 2. ESM-IF FC ranking (direct by score)
    if 'esmif_logratio' in result_df.columns:
        print("Ranking by ESM-IF FC...")
        result_df = rank_esmif_fc(result_df)

    # 3. ESM Z-score ranking
    print(f"Calculating ESM z-scores (grouped by {args.z_score_grouping})...")
    esm_z_df = calculate_z_scores(
        result_df,
        score_col='esm_average_logratio',
        grouping_col=args.z_score_grouping,
        z_col_name='esm_z_score'
    )
    if len(esm_z_df) > 0:
        esm_z_df = rank_by_zscore(esm_z_df, 'esm_z_score', 'esm_z_rank')
        # Merge z-scores back to main df
        result_df = result_df.merge(
            esm_z_df[['mutations', 'esm_z_score', 'esm_z_rank']],
            on='mutations',
            how='left'
        )
    else:
        print("  Warning: No groups with >= 5 samples for ESM z-score calculation")

    # 4. ESM-IF Z-score ranking
    if 'esmif_logratio' in result_df.columns:
        print(f"Calculating ESM-IF z-scores (grouped by {args.z_score_grouping})...")
        esmif_z_df = calculate_z_scores(
            result_df,
            score_col='esmif_logratio',
            grouping_col=args.z_score_grouping,
            z_col_name='esmif_z_score'
        )
        if len(esmif_z_df) > 0:
            esmif_z_df = rank_by_zscore(esmif_z_df, 'esmif_z_score', 'esmif_z_rank')
            # Merge z-scores back to main df
            result_df = result_df.merge(
                esmif_z_df[['mutations', 'esmif_z_score', 'esmif_z_rank']],
                on='mutations',
                how='left'
            )
        else:
            print("  Warning: No groups with >= 5 samples for ESM-IF z-score calculation")

    # Sort output by ESM FC rank (primary ranking)
    result_df = result_df.sort_values(by='esm_fc_rank')

    # Save single mutant results
    result_df.to_csv(output_file, index=False)
    print(f"\nSaved single mutant predictions to: {output_file}")

    # ===========================================
    # SAMPLING: Generate sampled variant lists
    # ===========================================
    if args.n_variants:
        print(f"\n=== Sampling Top {args.n_variants} Variants ===")
        wt_dir = os.path.dirname(output_file)
        if not wt_dir:
            wt_dir = '.'

        sampling_results = {}

        # ESM FC sampling (both methods)
        print("Sampling from ESM FC ranking...")
        esm_fc_top_n = sample_top_n(result_df, args.n_variants, 'esm_fc_rank')
        esm_fc_pos_excl = sample_position_exclusive(
            result_df, args.n_variants, 'esm_fc_rank', args.excluded_positions
        )
        sampling_results['esm_fc_top_n'] = esm_fc_top_n
        sampling_results['esm_fc_position_exclusive'] = esm_fc_pos_excl

        # ESM-IF FC sampling
        if 'esmif_fc_rank' in result_df.columns:
            print("Sampling from ESM-IF FC ranking...")
            esmif_fc_top_n = sample_top_n(result_df, args.n_variants, 'esmif_fc_rank')
            esmif_fc_pos_excl = sample_position_exclusive(
                result_df, args.n_variants, 'esmif_fc_rank', args.excluded_positions
            )
            sampling_results['esmif_fc_top_n'] = esmif_fc_top_n
            sampling_results['esmif_fc_position_exclusive'] = esmif_fc_pos_excl

        # ESM Z-score sampling
        if 'esm_z_rank' in result_df.columns:
            print("Sampling from ESM Z-score ranking...")
            esm_z_valid = result_df[result_df['esm_z_rank'].notna()].copy()
            esm_z_top_n = sample_top_n(esm_z_valid, args.n_variants, 'esm_z_rank')
            esm_z_pos_excl = sample_position_exclusive(
                esm_z_valid, args.n_variants, 'esm_z_rank', args.excluded_positions
            )
            sampling_results['esm_z_top_n'] = esm_z_top_n
            sampling_results['esm_z_position_exclusive'] = esm_z_pos_excl

        # ESM-IF Z-score sampling
        if 'esmif_z_rank' in result_df.columns:
            print("Sampling from ESM-IF Z-score ranking...")
            esmif_z_valid = result_df[result_df['esmif_z_rank'].notna()].copy()
            esmif_z_top_n = sample_top_n(esmif_z_valid, args.n_variants, 'esmif_z_rank')
            esmif_z_pos_excl = sample_position_exclusive(
                esmif_z_valid, args.n_variants, 'esmif_z_rank', args.excluded_positions
            )
            sampling_results['esmif_z_top_n'] = esmif_z_top_n
            sampling_results['esmif_z_position_exclusive'] = esmif_z_pos_excl

        # Save sampled variants
        sampled_output = output_file.replace('.csv', '_sampled.csv')

        # Create combined sampling summary
        all_sampled_mutations = set()
        sampling_flags = []

        for method_name, sampled_df in sampling_results.items():
            if len(sampled_df) > 0:
                for mut in sampled_df['mutations'].values:
                    all_sampled_mutations.add(mut)

        # Build summary dataframe with sampling flags
        summary_rows = []
        for mut in all_sampled_mutations:
            row = {'mutations': mut}
            for method_name, sampled_df in sampling_results.items():
                row[method_name] = 1 if mut in sampled_df['mutations'].values else 0
            summary_rows.append(row)

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            # Merge with scores from result_df
            score_cols = ['mutations', 'esm_average_logratio', 'esm_total_pass', 'esm_fc_rank']
            if 'esmif_logratio' in result_df.columns:
                score_cols.extend(['esmif_logratio', 'esmif_fc_rank'])
            if 'esm_z_score' in result_df.columns:
                score_cols.extend(['esm_z_score', 'esm_z_rank'])
            if 'esmif_z_score' in result_df.columns:
                score_cols.extend(['esmif_z_score', 'esmif_z_rank'])

            summary_df = summary_df.merge(
                result_df[score_cols].drop_duplicates(),
                on='mutations',
                how='left'
            )
            summary_df.to_csv(sampled_output, index=False)
            print(f"Saved sampled variants to: {sampled_output}")

            # Print sampling summary
            print(f"\nSampling Summary:")
            print(f"  Total unique mutations sampled: {len(all_sampled_mutations)}")
            for method_name, sampled_df in sampling_results.items():
                print(f"  {method_name}: {len(sampled_df)} mutations")

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

            # Add derived columns for double mutants (for z-score grouping)
            # Use the first mutation's substitution type for grouping
            double_df['aa_mutation'] = double_df['single1'].apply(lambda x: x[-1])
            double_df['aa_substitution_type'] = double_df['single1'].apply(lambda x: f'{x[0]}-{x[-1]}')

            # ===========================================
            # RANKING: Apply all ranking strategies to double mutants
            # ===========================================
            print("\n=== Applying Ranking Strategies to Double Mutants ===")

            # 1. ESM FC ranking (two-tier: consensus then magnitude)
            print("Ranking double mutants by ESM FC (two-tier)...")
            double_df = rank_esm_fc(double_df)

            # 2. ESM-IF FC ranking (direct by score)
            if 'esmif_logratio' in double_df.columns:
                print("Ranking double mutants by ESM-IF FC...")
                double_df = rank_esmif_fc(double_df)

            # 3. ESM Z-score ranking
            print(f"Calculating ESM z-scores for double mutants (grouped by {args.z_score_grouping})...")
            double_esm_z_df = calculate_z_scores(
                double_df,
                score_col='esm_average_logratio',
                grouping_col=args.z_score_grouping,
                z_col_name='esm_z_score'
            )
            if len(double_esm_z_df) > 0:
                double_esm_z_df = rank_by_zscore(double_esm_z_df, 'esm_z_score', 'esm_z_rank')
                double_df = double_df.merge(
                    double_esm_z_df[['mutations', 'esm_z_score', 'esm_z_rank']],
                    on='mutations',
                    how='left'
                )
            else:
                print("  Warning: No groups with >= 5 samples for double mutant ESM z-score calculation")

            # 4. ESM-IF Z-score ranking
            if 'esmif_logratio' in double_df.columns:
                print(f"Calculating ESM-IF z-scores for double mutants (grouped by {args.z_score_grouping})...")
                double_esmif_z_df = calculate_z_scores(
                    double_df,
                    score_col='esmif_logratio',
                    grouping_col=args.z_score_grouping,
                    z_col_name='esmif_z_score'
                )
                if len(double_esmif_z_df) > 0:
                    double_esmif_z_df = rank_by_zscore(double_esmif_z_df, 'esmif_z_score', 'esmif_z_rank')
                    double_df = double_df.merge(
                        double_esmif_z_df[['mutations', 'esmif_z_score', 'esmif_z_rank']],
                        on='mutations',
                        how='left'
                    )
                else:
                    print("  Warning: No groups with >= 5 samples for double mutant ESM-IF z-score calculation")

            # Sort output by ESM FC rank (primary ranking)
            double_df = double_df.sort_values(by='esm_fc_rank')

            # Save double mutant results
            double_output = output_file.replace('.csv', '_doubles.csv')
            double_df.to_csv(double_output, index=False)
            print(f"\nSaved double mutant predictions to: {double_output}")

            # ===========================================
            # SAMPLING: Generate sampled double mutant lists
            # ===========================================
            if args.n_variants:
                print(f"\n=== Sampling Top {args.n_variants} Double Mutants ===")

                double_sampling_results = {}

                # ESM FC sampling (top n only, no position-exclusive for doubles)
                print("Sampling from ESM FC ranking...")
                double_esm_fc_top_n = sample_top_n(double_df, args.n_variants, 'esm_fc_rank')
                double_sampling_results['esm_fc_top_n'] = double_esm_fc_top_n

                # ESM-IF FC sampling
                if 'esmif_fc_rank' in double_df.columns:
                    print("Sampling from ESM-IF FC ranking...")
                    double_esmif_fc_top_n = sample_top_n(double_df, args.n_variants, 'esmif_fc_rank')
                    double_sampling_results['esmif_fc_top_n'] = double_esmif_fc_top_n

                # ESM Z-score sampling
                if 'esm_z_rank' in double_df.columns:
                    print("Sampling from ESM Z-score ranking...")
                    double_esm_z_valid = double_df[double_df['esm_z_rank'].notna()].copy()
                    if len(double_esm_z_valid) > 0:
                        double_esm_z_top_n = sample_top_n(double_esm_z_valid, args.n_variants, 'esm_z_rank')
                        double_sampling_results['esm_z_top_n'] = double_esm_z_top_n

                # ESM-IF Z-score sampling
                if 'esmif_z_rank' in double_df.columns:
                    print("Sampling from ESM-IF Z-score ranking...")
                    double_esmif_z_valid = double_df[double_df['esmif_z_rank'].notna()].copy()
                    if len(double_esmif_z_valid) > 0:
                        double_esmif_z_top_n = sample_top_n(double_esmif_z_valid, args.n_variants, 'esmif_z_rank')
                        double_sampling_results['esmif_z_top_n'] = double_esmif_z_top_n

                # Save sampled double mutants
                double_sampled_output = output_file.replace('.csv', '_doubles_sampled.csv')

                # Create combined sampling summary for doubles
                all_double_sampled = set()
                for method_name, sampled_df in double_sampling_results.items():
                    if len(sampled_df) > 0:
                        for mut in sampled_df['mutations'].values:
                            all_double_sampled.add(mut)

                # Build summary dataframe with sampling flags
                double_summary_rows = []
                for mut in all_double_sampled:
                    row = {'mutations': mut}
                    for method_name, sampled_df in double_sampling_results.items():
                        row[method_name] = 1 if mut in sampled_df['mutations'].values else 0
                    double_summary_rows.append(row)

                if double_summary_rows:
                    double_summary_df = pd.DataFrame(double_summary_rows)
                    # Merge with scores from double_df
                    double_score_cols = ['mutations', 'single1', 'single2', 'esm_average_logratio', 'esm_total_pass', 'esm_fc_rank']
                    if 'esmif_logratio' in double_df.columns:
                        double_score_cols.extend(['esmif_logratio', 'esmif_fc_rank'])
                    if 'esm_z_score' in double_df.columns:
                        double_score_cols.extend(['esm_z_score', 'esm_z_rank'])
                    if 'esmif_z_score' in double_df.columns:
                        double_score_cols.extend(['esmif_z_score', 'esmif_z_rank'])

                    double_summary_df = double_summary_df.merge(
                        double_df[double_score_cols].drop_duplicates(),
                        on='mutations',
                        how='left'
                    )
                    double_summary_df.to_csv(double_sampled_output, index=False)
                    print(f"Saved sampled double mutants to: {double_sampled_output}")

                    # Print sampling summary
                    print(f"\nDouble Mutant Sampling Summary:")
                    print(f"  Total unique double mutants sampled: {len(all_double_sampled)}")
                    for method_name, sampled_df in double_sampling_results.items():
                        print(f"  {method_name}: {len(sampled_df)} mutations")

    # Print summary statistics
    print("\n=== Single Mutant Summary ===")
    print(f"Total single mutations analyzed: {len(result_df)}")

    # ESM statistics
    if 'esm_average_logratio' in result_df.columns:
        avg_score = result_df['esm_average_logratio'].mean()
        beneficial = (result_df['esm_average_logratio'] > 0).sum()
        print(f"\nESM predictions (non structure-informed):")
        print(f"  Mean log-ratio: {avg_score:.4f}")
        print(f"  Beneficial mutations (y' > 1): {beneficial}")

    if 'esm_total_pass' in result_df.columns:
        max_pass = result_df['esm_total_pass'].max()
        all_pass = (result_df['esm_total_pass'] == max_pass).sum()
        consensus = (result_df['esm_total_pass'] > 0).sum()
        print(f"  Mutations with f(y') > 0 (model consensus): {consensus}")
        print(f"  Mutations passing all {int(max_pass)} models: {all_pass}")

    # ESM-IF statistics
    if 'esmif_logratio' in result_df.columns:
        avg_esmif = result_df['esmif_logratio'].mean()
        beneficial_esmif = (result_df['esmif_logratio'] > 0).sum()
        print(f"\nESM-IF predictions (structure-informed):")
        print(f"  Mean log-ratio: {avg_esmif:.4f}")
        print(f"  Beneficial mutations (y' > 1): {beneficial_esmif}")

    # Top single mutations by ESM FC ranking
    print(f"\nTop 10 single mutations by ESM FC ranking (two-tier):")
    top_cols = ['mutations', 'esm_fc_rank', 'esm_total_pass', 'esm_average_logratio']
    if 'esmif_logratio' in result_df.columns:
        top_cols.append('esmif_logratio')

    top_df = result_df.nsmallest(10, 'esm_fc_rank')[top_cols]
    print(top_df.to_string(index=False))

    # Top by ESM-IF if available
    if 'esmif_fc_rank' in result_df.columns:
        print(f"\nTop 10 single mutations by ESM-IF FC ranking:")
        esmif_cols = ['mutations', 'esmif_fc_rank', 'esmif_logratio', 'esm_total_pass', 'esm_average_logratio']
        top_esmif_df = result_df.nsmallest(10, 'esmif_fc_rank')[esmif_cols]
        print(top_esmif_df.to_string(index=False))

    # Double mutant statistics
    if double_df is not None and len(double_df) > 0:
        print(f"\n=== Double Mutant Summary ===")
        print(f"Total double mutants scored: {len(double_df)}")

        if 'esm_average_logratio' in double_df.columns:
            avg_double = double_df['esm_average_logratio'].mean()
            beneficial_double = (double_df['esm_average_logratio'] > 0).sum()
            print(f"\nESM predictions (non structure-informed):")
            print(f"  Mean log-ratio: {avg_double:.4f}")
            print(f"  Beneficial double mutants (y' > 1): {beneficial_double}")

        if 'esm_total_pass' in double_df.columns:
            max_pass = double_df['esm_total_pass'].max()
            all_pass_double = (double_df['esm_total_pass'] == max_pass).sum()
            consensus_double = (double_df['esm_total_pass'] > 0).sum()
            print(f"  Double mutants with f(y') > 0 (model consensus): {consensus_double}")
            print(f"  Double mutants passing all {int(max_pass)} models: {all_pass_double}")

        if 'esmif_logratio' in double_df.columns:
            avg_esmif_double = double_df['esmif_logratio'].mean()
            beneficial_esmif_double = (double_df['esmif_logratio'] > 0).sum()
            print(f"\nESM-IF predictions (structure-informed):")
            print(f"  Mean log-ratio: {avg_esmif_double:.4f}")
            print(f"  Beneficial double mutants (y' > 1): {beneficial_esmif_double}")

        # Top double mutants by ESM FC ranking
        print(f"\nTop 10 double mutants by ESM FC ranking (two-tier):")
        double_top_cols = ['mutations', 'esm_fc_rank', 'esm_total_pass', 'esm_average_logratio']
        if 'esmif_logratio' in double_df.columns:
            double_top_cols.append('esmif_logratio')

        top_double_df = double_df.nsmallest(10, 'esm_fc_rank')[double_top_cols]
        print(top_double_df.to_string(index=False))

        # Top by ESM-IF if available
        if 'esmif_fc_rank' in double_df.columns:
            print(f"\nTop 10 double mutants by ESM-IF FC ranking:")
            esmif_double_cols = ['mutations', 'esmif_fc_rank', 'esmif_logratio', 'esm_total_pass', 'esm_average_logratio']
            top_esmif_double_df = double_df.nsmallest(10, 'esmif_fc_rank')[esmif_double_cols]
            print(top_esmif_double_df.to_string(index=False))


if __name__ == '__main__':
    main()
