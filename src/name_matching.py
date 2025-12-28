"""
Player name matching utilities using fuzzy string matching.
"""

import pandas as pd
from rapidfuzz import fuzz
from typing import Tuple


def normalize_name(name: str) -> str:
    """
    Normalize player name for matching.

    Args:
        name: Original player name

    Returns:
        Normalized name (lowercase, no dots, single spaces)
    """
    if pd.isna(name):
        return ""
    name = str(name).strip()
    name = name.replace(".", "")
    name = ' '.join(name.split())
    return name.lower()


def match_players_in_match(
    df_match: pd.DataFrame,
    df_reference: pd.DataFrame
) -> pd.DataFrame:
    """
    Match players within a single match using optimal assignment based on name similarity.

    Args:
        df_match: DataFrame with player data to match (must have 'player_name' column)
        df_reference: DataFrame with reference data (must have 'player_name' and target columns)

    Returns:
        DataFrame with matched data appended
    """
    df_match = df_match.copy().reset_index(drop=True)
    df_reference = df_reference.copy().reset_index(drop=True)

    df_match['player_name_norm'] = df_match['player_name'].apply(normalize_name)
    df_reference['player_name_norm'] = df_reference['player_name'].apply(normalize_name)

    # Compute all pairwise similarities
    matches = []
    for i, row_match in df_match.iterrows():
        for j, row_ref in df_reference.iterrows():
            similarity = fuzz.ratio(
                row_match['player_name_norm'],
                row_ref['player_name_norm']
            )
            matches.append({
                'match_idx': i,
                'ref_idx': j,
                'similarity': similarity,
                'match_name': row_match['player_name'],
                'ref_name': row_ref['player_name'],
                'ref_data': row_ref.to_dict()
            })

    matches_df = pd.DataFrame(matches)
    matches_df = matches_df.sort_values('similarity', ascending=False)

    # Greedy optimal assignment
    used_match = set()
    used_ref = set()
    final_matches = []

    for _, match in matches_df.iterrows():
        match_idx = match['match_idx']
        ref_idx = match['ref_idx']

        if match_idx not in used_match and ref_idx not in used_ref:
            used_match.add(match_idx)
            used_ref.add(ref_idx)
            final_matches.append({
                'match_idx': match_idx,
                'match_confidence': match['similarity'],
                'matched_with': match['ref_name'],
                'ref_data': match['ref_data']
            })

    # Build result
    result_rows = []
    for i, row in df_match.iterrows():
        matched = next((m for m in final_matches if m['match_idx'] == i), None)
        row_dict = row.to_dict()

        if matched:
            # Add reference data
            for key, value in matched['ref_data'].items():
                if key not in ['player_name', 'player_name_norm']:
                    row_dict[key] = value
            row_dict['match_confidence'] = matched['match_confidence']
            row_dict['matched_with'] = matched['matched_with']
        else:
            row_dict['match_confidence'] = 0
            row_dict['matched_with'] = None

        result_rows.append(row_dict)

    result_df = pd.DataFrame(result_rows)

    if 'player_name_norm' in result_df.columns:
        result_df = result_df.drop(columns=['player_name_norm'])

    return result_df


def merge_with_fuzzy_matching(
    df_main: pd.DataFrame,
    df_reference: pd.DataFrame,
    match_id_col: str = 'match_id',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Merge two DataFrames using fuzzy name matching within each match.

    Args:
        df_main: Main DataFrame (must have match_id_col and 'player_name')
        df_reference: Reference DataFrame (must have match_id_col and 'player_name')
        match_id_col: Name of the match ID column
        verbose: Print matching statistics

    Returns:
        Merged DataFrame with reference data appended
    """
    df_main = df_main.copy()
    df_reference = df_reference.copy()

    all_results = []

    for match_id in df_main[match_id_col].unique():
        if verbose:
            print(f"Processing match {match_id}...")

        df_match = df_main[df_main[match_id_col] == match_id]
        df_ref_match = df_reference[df_reference[match_id_col] == match_id]

        if df_ref_match.empty:
            if verbose:
                print(f"  Warning: Match {match_id} not found in df_reference")

            # Add empty reference columns
            for _, row in df_match.iterrows():
                row_dict = row.to_dict()
                row_dict['match_confidence'] = 0
                row_dict['matched_with'] = None
                all_results.append(row_dict)
        else:
            result = match_players_in_match(df_match, df_ref_match)
            all_results.append(result)

            if verbose:
                matched = result.get('match_confidence', pd.Series([0])).gt(0).sum()
                total = len(result)
                avg_conf = result[result.get('match_confidence', 0) > 0].get('match_confidence', pd.Series([0])).mean()
                print(f"  Matched: {matched}/{total} players (avg confidence: {avg_conf:.1f})")

    df_result = pd.concat(all_results, ignore_index=True)

    return df_result
