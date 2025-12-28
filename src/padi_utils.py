"""
Utility functions for PADI (Phase-Adjusted Decision Index) calculation.

This module contains the core functions for:
- Data preparation and filtering
- Phase association
- Phase weight estimation via logistic regression
- PADI value computation
- Player-match aggregation
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def filter_pass_decisions(df_events: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and prepare pass events from dynamic events data.

    Extracts player_possession events that end in passes, computes timestamps,
    and prepares passer information.

    Args:
        df_events: DataFrame with dynamic events

    Returns:
        DataFrame with valid pass events
    """
    df_pp = df_events[df_events['event_type'] == 'player_possession'].copy()
    df_pass = df_pp[df_pp['end_type'] == 'pass'].copy()

    def parse_timestamp_to_seconds(row):
        time_str = str(row.get('time_end') or row.get('time_start', '0:0'))
        try:
            parts = time_str.split(':')
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        except:
            pass
        return np.nan

    df_pass['pass_time_seconds'] = df_pass.apply(parse_timestamp_to_seconds, axis=1)
    df_pass['pass_time_seconds_adjusted'] = df_pass.apply(
        lambda row: row['pass_time_seconds'] + (45 * 60 if row.get('period') == 2 else 0), axis=1
    )
    df_pass = df_pass[df_pass['pass_time_seconds'].notna()].copy()

    df_pass['passer_id'] = df_pass['player_id']
    df_pass['passer_name'] = df_pass['player_name']
    df_pass['passer_position'] = df_pass['player_position']

    print(f"{len(df_pass):,} valid passes ({df_pass['passer_id'].nunique()} players)")
    return df_pass


def get_passing_options(df_events: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and filter passing option events.

    Retrieves passing_option events with valid xThreat and xPass values.

    Args:
        df_events: DataFrame with dynamic events

    Returns:
        DataFrame with valid passing options
    """
    df_po = df_events[df_events['event_type'] == 'passing_option'].copy()

    cols_keep = [
        'event_id', 'match_id', 'period', 'associated_player_possession_event_id',
        'player_id', 'player_name', 'player_position',
        'xthreat', 'xpass_completion',
        'dangerous', 'difficult_pass_target',
        'n_opponents_ahead_pass_reception', 'n_opponents_bypassed', 'n_simultaneous_passing_options'
    ]

    cols_available = [c for c in cols_keep if c in df_po.columns]
    df_po = df_po[cols_available].copy()
    df_po = df_po[(df_po['xthreat'].notna()) & (df_po['xpass_completion'].notna())].copy()

    print(f"{len(df_po):,} valid passing_options ({df_po['associated_player_possession_event_id'].nunique():,} possessions)")
    return df_po


def attach_phases(df_pass: pd.DataFrame, df_phases: pd.DataFrame) -> pd.DataFrame:
    """
    Associate game phases to pass events based on temporal matching.

    Matches each pass to its corresponding phase of play based on match_id,
    period, and timestamp.

    Args:
        df_pass: DataFrame with pass events
        df_phases: DataFrame with phase of play data

    Returns:
        DataFrame with phase_name column added
    """
    def parse_phase_timestamp(time_str):
        if pd.isna(time_str):
            return np.nan
        try:
            parts = str(time_str).split(':')
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        except:
            pass
        return np.nan

    df_phases = df_phases.copy()
    df_phases['phase_start_seconds'] = df_phases['time_start'].apply(parse_phase_timestamp)
    df_phases['phase_end_seconds'] = df_phases['time_end'].apply(parse_phase_timestamp)
    df_phases['phase_start_seconds_adjusted'] = df_phases.apply(
        lambda row: row['phase_start_seconds'] + (45 * 60 if row.get('period') == 2 else 0), axis=1
    )
    df_phases['phase_end_seconds_adjusted'] = df_phases.apply(
        lambda row: row['phase_end_seconds'] + (45 * 60 if row.get('period') == 2 else 0), axis=1
    )
    df_phases = df_phases[df_phases['phase_start_seconds'].notna() & df_phases['phase_end_seconds'].notna()].copy()

    df_pass_with_phase = df_pass.copy()
    df_pass_with_phase['phase_name'] = None

    phases_grouped = df_phases.groupby(['match_id', 'period'])

    for idx, pass_row in df_pass_with_phase.iterrows():
        match_id = pass_row['match_id']
        period = pass_row['period']
        pass_time = pass_row['pass_time_seconds_adjusted']

        try:
            phases_subset = phases_grouped.get_group((match_id, period))
            matching_phases = phases_subset[
                (phases_subset['phase_start_seconds_adjusted'] <= pass_time) &
                (phases_subset['phase_end_seconds_adjusted'] >= pass_time)
            ]

            if len(matching_phases) > 0:
                df_pass_with_phase.at[idx, 'phase_name'] = matching_phases.iloc[0]['team_in_possession_phase_type']
        except KeyError:
            continue

    df_pass_with_phase['phase_name'] = df_pass_with_phase['phase_name'].fillna('default')

    print(f"Phases attached")
    print(f"Distribution: {df_pass_with_phase['phase_name'].value_counts().to_dict()}")
    return df_pass_with_phase


def estimate_phase_weights(
    df_pass_with_phase: pd.DataFrame,
    df_passing_options: pd.DataFrame
) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
    """
    Estimate phase-specific weights using logistic regression on actual player choices.

    For each phase, models the probability of choosing an option based on normalized
    threat (S_T) and safety (S_P) scores, controlling for defensive pressure and risk.
    Weights are derived from the regression coefficients.

    Args:
        df_pass_with_phase: DataFrame with passes and phase labels
        df_passing_options: DataFrame with passing options

    Returns:
        Tuple of (weights_dict, diagnostics_df) where weights_dict maps phase names
        to {'wT': float, 'wP': float} and diagnostics_df contains model fitting info
    """
    weights_by_phase = {}
    diagnostics = []

    df_pass_filtered = df_pass_with_phase[df_pass_with_phase['passer_position'] != 'GK'].copy()

    for phase_name in df_pass_filtered['phase_name'].unique():
        phase_pass = df_pass_filtered[df_pass_filtered['phase_name'] == phase_name]
        data = []

        for _, pass_row in phase_pass.iterrows():
            pp_event_id = pass_row['event_id']
            chosen_po_event_id = pass_row.get('targeted_passing_option_event_id')

            options = df_passing_options[
                df_passing_options['associated_player_possession_event_id'] == pp_event_id
            ].copy()

            if len(options) < 2:
                continue

            N = len(options)
            options['rank_xT'] = options['xthreat'].rank(method='min', ascending=False)
            options['rank_xP'] = options['xpass_completion'].rank(method='min', ascending=False)

            for _, opt in options.iterrows():
                S_T = (N - opt['rank_xT'] + 1) / N
                S_P = (N - opt['rank_xP'] + 1) / N
                chosen = 1 if opt['event_id'] == chosen_po_event_id else 0

                data.append({
                    'S_T': S_T,
                    'S_P': S_P,
                    'chosen': chosen,
                    'dangerous': opt.get('dangerous'),
                    'difficult_pass_target': opt.get('difficult_pass_target'),
                    'n_opponents_ahead_pass_reception': opt.get('n_opponents_ahead_pass_reception'),
                    'n_opponents_bypassed': opt.get('n_opponents_bypassed'),
                    'n_simultaneous_passing_options': opt.get('n_simultaneous_passing_options')
                })

        df_model = pd.DataFrame(data)

        if len(df_model) < 10 or df_model['chosen'].sum() < 5:
            weights_by_phase[phase_name] = {'wT': 0.5, 'wP': 0.5}
            diagnostics.append({
                'phase': phase_name, 'n_options': len(df_model), 'n_chosen': df_model['chosen'].sum() if len(df_model) > 0 else 0,
                'wT': 0.5, 'wP': 0.5, 'coef_T': np.nan, 'coef_P': np.nan, 'status': 'insufficient_data'
            })
            continue

        df_model['dangerous'] = (
            df_model['dangerous'].fillna(False).astype(str).str.lower().isin(['true', '1']).astype(int)
        )
        df_model['difficult_pass_target'] = (
            df_model['difficult_pass_target'].fillna(False).astype(str).str.lower().isin(['true', '1']).astype(int)
        )

        num_cols = [
            'S_T', 'S_P',
            'n_opponents_ahead_pass_reception', 'n_opponents_bypassed', 'n_simultaneous_passing_options',
            'dangerous', 'difficult_pass_target'
        ]

        for col in num_cols:
            df_model[col] = pd.to_numeric(df_model[col], errors='coerce')

        for col in num_cols:
            if df_model[col].isna().any():
                fill_value = df_model[col].median()
                if pd.isna(fill_value):
                    fill_value = 0.0
                df_model[col] = df_model[col].fillna(fill_value)

        feature_cols = num_cols

        X = df_model[num_cols].values
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = df_model['chosen'].values

        try:
            model = LogisticRegression(penalty=None, max_iter=1000, solver='lbfgs')
            model.fit(X, y)

            coef = model.coef_[0]
            coef_T = coef[feature_cols.index('S_T')]
            coef_P = coef[feature_cols.index('S_P')]

            total = abs(coef_T) + abs(coef_P)
            wT = abs(coef_T) / total if total > 0 else 0.5
            wP = abs(coef_P) / total if total > 0 else 0.5

            weights_by_phase[phase_name] = {'wT': wT, 'wP': wP}
            diagnostics.append({
                'phase': phase_name, 'n_options': len(df_model), 'n_chosen': df_model['chosen'].sum(),
                'wT': wT, 'wP': wP, 'coef_T': coef_T, 'coef_P': coef_P, 'status': 'success'
            })
        except Exception as e:
            weights_by_phase[phase_name] = {'wT': 0.5, 'wP': 0.5}
            diagnostics.append({
                'phase': phase_name, 'n_options': len(df_model), 'n_chosen': df_model['chosen'].sum(),
                'wT': 0.5, 'wP': 0.5, 'coef_T': np.nan, 'coef_P': np.nan, 'status': f'error'
            })

    return weights_by_phase, pd.DataFrame(diagnostics)


def compute_padi(
    df_pass: pd.DataFrame,
    df_passing_options: pd.DataFrame,
    phase_weights: Dict
) -> pd.DataFrame:
    """
    Calculate PADI values for each passing decision.

    For each pass, computes threat (S_T) and safety (S_P) scores based on the rank
    of the chosen option, applies phase-specific weights, and calculates the final
    PADI value using the formula: PADI = (Q × xT_chosen)^0.25

    Args:
        df_pass: DataFrame with pass events and phases
        df_passing_options: DataFrame with passing options
        phase_weights: Dictionary mapping phase names to {'wT': float, 'wP': float}

    Returns:
        DataFrame with PADI values for each decision
    """
    results = []
    df_pass = df_pass[df_pass['passer_position'] != 'GK'].copy()

    for _, pass_row in df_pass.iterrows():
        pp_event_id = pass_row['event_id']
        phase_name = pass_row['phase_name']

        options = df_passing_options[
            df_passing_options['associated_player_possession_event_id'] == pp_event_id
        ].copy()

        if len(options) == 0:
            continue

        chosen_po_event_id = pass_row.get('targeted_passing_option_event_id')
        chosen_option = options[options['event_id'] == chosen_po_event_id]

        if len(chosen_option) == 0:
            continue

        chosen_xt = chosen_option.iloc[0]['xthreat']
        chosen_xp = chosen_option.iloc[0]['xpass_completion']

        N = len(options)
        options['rank_xT'] = options['xthreat'].rank(method='min', ascending=False)
        options['rank_xP'] = options['xpass_completion'].rank(method='min', ascending=False)

        r_T_chosen = options[options['event_id'] == chosen_po_event_id]['rank_xT'].iloc[0]
        r_P_chosen = options[options['event_id'] == chosen_po_event_id]['rank_xP'].iloc[0]

        S_T = (N - r_T_chosen + 1) / N
        S_P = (N - r_P_chosen + 1) / N

        weights = phase_weights.get(phase_name, {'wT': 0.5, 'wP': 0.5})
        wT = weights['wT']
        wP = weights['wP']

        Q = (S_T + S_P) / (S_T / wT + S_P / wP)

        results.append({
            'match_id': pass_row['match_id'],
            'period': pass_row['period'],
            'phase_name': phase_name,
            'passer_id': pass_row['passer_id'],
            'passer_name': pass_row['passer_name'],
            'passer_position': pass_row['passer_position'],
            'xT_chosen': chosen_xt,
            'xP_chosen': chosen_xp,
            'PADI_value': np.power(Q * chosen_xt, 0.25),
            'n_options': N
        })

    return pd.DataFrame(results)


def load_player_minutes(raw_data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load player minutes played from match JSON files.

    Extracts playing time information for each player-match combination from
    SkillCorner match metadata files.

    Args:
        raw_data_dir: Path to raw data directory

    Returns:
        DataFrame with player_id, match_id, minutes_played, and player_position
    """
    match_files = sorted(Path(raw_data_dir).glob("match/*_match.json"))
    all_player_minutes = []

    for match_file in match_files:
        match_id = int(match_file.stem.split('_')[0])

        with open(match_file, 'r', encoding='utf-8') as f:
            match_data = json.load(f)

        for player in match_data.get('players', []):
            playing_time = player.get('playing_time') or {}
            total_time = playing_time.get('total') or {}

            minutes_played = total_time.get('minutes_played_regular_time')
            if not minutes_played:
                minutes_played = total_time.get('minutes_played', 0.0)

            player_role = player.get('player_role') or {}
            position = player_role.get('acronym', 'Unknown')

            all_player_minutes.append({
                'player_id': player.get('id'),
                'player_name': player.get('short_name', ''),
                'match_id': match_id,
                'minutes_played': float(minutes_played or 0.0),
                'player_position': position
            })

    df_minutes = pd.DataFrame(all_player_minutes)
    print(f"✓ Minutes loaded: {len(df_minutes):,} player-match combinations")
    return df_minutes


def aggregate_player_match(
    df_decisions: pd.DataFrame,
    df_minutes: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate PADI values by player-match and merge with minutes played.

    Computes median PADI value and pass count for each player-match combination,
    then merges with playing time information.

    Args:
        df_decisions: DataFrame with PADI values for each decision
        df_minutes: DataFrame with player minutes played

    Returns:
        DataFrame with aggregated player-match statistics
    """
    df_agg = df_decisions.groupby(['passer_id', 'passer_name', 'match_id']).agg(
        n_pass=('xT_chosen', 'count'),
        PADI_value_median=('PADI_value', 'median'),
        position=('passer_position', lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]),
    ).reset_index()

    df_agg = df_agg.merge(
        df_minutes[['player_id', 'match_id', 'minutes_played']],
        left_on=['passer_id', 'match_id'],
        right_on=['player_id', 'match_id'],
        how='left'
    ).drop(columns=['player_id'])

    df_agg['minutes_played'] = df_agg['minutes_played'].fillna(0.0)

    df_agg = df_agg.rename(columns={
        'passer_id': 'player_id',
        'passer_name': 'player_name',
        'position': 'player_position'
    })

    return df_agg
