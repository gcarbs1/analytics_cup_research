"""
process_data.py
Author: Gabriel Carbinatto (gabrielcarbinatto@usp.br)

"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple


class MatchDataProcessor:
    def __init__(self, raw_data_dir: str = "data/raw", processed_data_dir: str = "data/processed"):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
    def process_match(self, match_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        print(f"Processing match {match_id}...")
        df_match, df_players_info = self._process_match_and_players(match_id)
        df_tracking = self._process_tracking(match_id, df_players_info)
        self._save_processed_data(match_id, df_match, df_players_info, df_tracking)
        print(f"Match {match_id} processed and saved.")
        return df_match, df_players_info, df_tracking
    
    def _process_match_and_players(self, match_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        match_path = self.raw_data_dir / "match" / f"{match_id}_match.json"
        with open(match_path, "r", encoding="utf-8") as f:
            match_data = json.load(f)
        
        df_match = pd.json_normalize(match_data, sep=".")
        
        cols_remove_match = [
            "date_time", "home_team_coach", "away_team_coach", "match_periods", "referees", "status",
            "pitch_length", "pitch_width", "stadium.id", "stadium.name", "stadium.city", "stadium.capacity",
            "home_team.name", "home_team_kit.id", "home_team_kit.team_id", "home_team_kit.season.id",
            "home_team_kit.season.start_year", "home_team_kit.season.end_year", "home_team_kit.season.name",
            "home_team_kit.name", "home_team_kit.jersey_color", "home_team_kit.number_color",
            "away_team.name", "away_team_kit.id", "away_team_kit.team_id", "away_team_kit.season.id",
            "away_team_kit.season.start_year", "away_team_kit.season.end_year", "away_team_kit.season.name",
            "away_team_kit.name", "away_team_kit.jersey_color", "away_team_kit.number_color",
            "home_team_playing_time.minutes_tip", "home_team_playing_time.minutes_otip",
            "away_team_playing_time.minutes_tip", "away_team_playing_time.minutes_otip",
            "competition_edition.competition.gender", "competition_edition.competition.age_group",
            "competition_edition.season.id", "competition_edition.season.start_year",
            "competition_edition.season.end_year", "competition_edition.name",
            "competition_round.id", "competition_round.name", "competition_round.round_number",
            "competition_round.potential_overtime", "ball.trackable_object", "players",
            "competition_edition.id", "competition_edition.competition.id"
        ]
        
        df_match = df_match.drop(
            columns=[c for c in cols_remove_match if c in df_match.columns], 
            errors="ignore"
        )
        
        df_match = df_match.rename(columns={
            "id": "sk_match_id",
            "home_team.id": "sk_home_team_id",
            "home_team.short_name": "home_team_name",
            "home_team.acronym": "home_team_abbrev_name",
            "away_team.id": "sk_away_team_id",
            "away_team.short_name": "away_team_name",
            "away_team.acronym": "away_team_abbrev_name",
            "competition_edition.competition.area": "sk_comp_code",
            "competition_edition.competition.name": "sk_comp_name",
            "competition_edition.season.name": "sk_season_name"
        })
        
        players_data = match_data.get("players", [])
        df_players_info = pd.json_normalize(players_data, sep=".")
        df_players_info["sk_match_id"] = match_id
        
        cols_remove_players = [
            "yellow_card", "red_card", "injured", "goal", "own_goal", "team_player_id",
            "first_name", "last_name", "birthday", "trackable_object", "gender", "player_role.id",
            "player_role.position_group", "player_role.name",
            "playing_time.total.minutes_tip", "playing_time.total.minutes_otip",
            "playing_time.total.start_frame", "playing_time.total.end_frame",
            "playing_time.total.minutes_played", "playing_time.by_period", "playing_time.total"
        ]
        
        df_players_info = df_players_info.drop(
            columns=[c for c in cols_remove_players if c in df_players_info.columns],
            errors="ignore"
        )
        
        df_players_info = df_players_info.rename(columns={
            "id": "sk_player_id",
            "team_id": "sk_team_id",
            "player_role.acronym": "sk_position"
        })
        
        if "playing_time.total.minutes_played_regular_time" in df_players_info.columns:
            df_players_info = df_players_info.rename(columns={
                "playing_time.total.minutes_played_regular_time": "minutes_played"
            })
        
        ordered_cols = ["sk_match_id", "sk_team_id", "sk_player_id"] + [
            c for c in df_players_info.columns 
            if c not in ["sk_match_id", "sk_team_id", "sk_player_id"]
        ]
        df_players_info = df_players_info[ordered_cols]
        
        return df_match, df_players_info
    
    def _process_tracking(
        self, 
        match_id: str, 
        df_players_info: pd.DataFrame
    ) -> pd.DataFrame:
        tracking_path = self.raw_data_dir / "tracking" / f"{match_id}_tracking_extrapolated.jsonl"

        with open(tracking_path, "r", encoding="utf-8") as f:
            tracking_lines = [json.loads(line) for line in f]
        df_tracking = pd.DataFrame(tracking_lines)

        df_tracking = df_tracking.sort_values("frame").reset_index(drop=True)

        df_tracking = df_tracking[
            df_tracking["timestamp"].notna()
            & df_tracking["timestamp"].astype(str).str.match(r"^\d{2}:\d{2}:\d{2}\.\d{2,6}$")
        ].copy()

        df_tracking["timestamp"] = pd.to_datetime(
            df_tracking["timestamp"].astype(str),
            format="%H:%M:%S.%f",
            errors="coerce"
        )

        df_tracking = df_tracking.dropna(subset=["timestamp"]).copy()

        df_tracking["timestamp"] = (
            df_tracking["timestamp"]
            .dt.strftime("%H:%M:%S.%f")
            .str[:-3]
        )

        df_tracking["period"] = df_tracking["period"].astype("Int64")
        df_tracking["frame"] = df_tracking["frame"].astype(int)
            
        df_tracking_clean = df_tracking[["frame", "timestamp", "period", "player_data"]].copy()
        df_tracking_clean.insert(0, "sk_match_id", match_id)
            
        df_tracking_clean = self._normalize_coordinates(df_tracking_clean)
        df_players_tracking_raw = self._expand_player_data(df_tracking_clean, df_players_info)
        df_players_tracking_final = self._orient_teams_using_first_frame(df_players_tracking_raw)

        return df_players_tracking_final
    
    def _normalize_coordinates(self, df_tracking: pd.DataFrame) -> pd.DataFrame:
        period_1_mask = df_tracking["period"] == 1
        
        if period_1_mask.any():
            last_period_1_idx = df_tracking[period_1_mask].index[-1]
            for idx in df_tracking.index[df_tracking.index > last_period_1_idx]:
                player_data = df_tracking.at[idx, "player_data"]
                if isinstance(player_data, list) and len(player_data) > 0:
                    for player in player_data:
                        if "x" in player and "y" in player:
                            player["x"] = -player["x"]
                            player["y"] = -player["y"]
        
        return df_tracking
    
    def _expand_player_data(
        self, 
        df_tracking: pd.DataFrame,
        df_players_info: pd.DataFrame
    ) -> pd.DataFrame:
        player_team_map = (
            df_players_info.set_index("sk_player_id")["sk_team_id"]
            .to_dict()
        )

        mask_valid = df_tracking["player_data"].apply(
            lambda x: isinstance(x, list) and len(x) > 0
        )
        player_rows = df_tracking.loc[mask_valid].reset_index(drop=True)
        
        expanded_rows = []
        for _, row in player_rows.iterrows():
            df_temp = pd.json_normalize(row["player_data"], sep=".")
            
            df_temp = df_temp.rename(columns={
                "x": "sk_pos_x",
                "y": "sk_pos_y",
                "player_id": "sk_player_id"
            })
            
            df_temp["sk_match_id"] = row["sk_match_id"]
            df_temp["frame"] = row["frame"]
            df_temp["timestamp"] = row["timestamp"]
            df_temp["period"] = row["period"]
            
            df_temp["sk_team_id"] = df_temp["sk_player_id"].map(player_team_map)
            
            expanded_rows.append(df_temp)
        
        df_players_full = pd.concat(expanded_rows, ignore_index=True)

        cols_order = [
            "sk_match_id", "frame", "timestamp", "period",
            "sk_player_id", "sk_team_id",
            "sk_pos_x", "sk_pos_y",
            "is_detected"
        ]
        df_players_full = df_players_full[cols_order]
        
        return df_players_full

    def _orient_teams_using_first_frame(
        self,
        df_players_tracking: pd.DataFrame
    ) -> pd.DataFrame:
        df_oriented = df_players_tracking.copy()

        first_frame = df_oriented["frame"].min()
        df_first = df_oriented[df_oriented["frame"] == first_frame]

        mean_x_by_team = (
            df_first.groupby("sk_team_id")["sk_pos_x"]
            .mean()
        )
        
        if len(mean_x_by_team) >= 2:
            team_right_to_left = mean_x_by_team.idxmax()

            mask_flip = df_oriented["sk_team_id"] == team_right_to_left
            df_oriented.loc[mask_flip, "sk_pos_x"] = -df_oriented.loc[mask_flip, "sk_pos_x"]
            df_oriented.loc[mask_flip, "sk_pos_y"] = -df_oriented.loc[mask_flip, "sk_pos_y"]

        df_oriented["sk_pos_x"] = df_oriented["sk_pos_x"].round(2)
        df_oriented["sk_pos_y"] = df_oriented["sk_pos_y"].round(2)

        return df_oriented
    
    def _save_processed_data(
        self, 
        match_id: str, 
        df_match: pd.DataFrame, 
        df_players_info: pd.DataFrame, 
        df_tracking: pd.DataFrame
    ) -> None:
        match_file = self.processed_data_dir / f"{match_id}_match_info.parquet"
        df_match.to_parquet(match_file, index=False)
        
        players_file = self.processed_data_dir / f"{match_id}_players_match_info.parquet"
        df_players_info.to_parquet(players_file, index=False)
        
        tracking_file = self.processed_data_dir / f"{match_id}_tracking.parquet"
        df_tracking.to_parquet(tracking_file, index=False)
    
    def process_all_matches(self, match_ids: list) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        print(f"Processing {len(match_ids)} matches in total...")
        results = {}
        for match_id in match_ids:
            try:
                results[match_id] = self.process_match(match_id)
            except Exception as e:
                print(f"Error while processing match {match_id}: {e}")
        print("All matches processed.")
        return results


def load_processed_match(match_id: str, processed_data_dir: str = "data/processed") -> Dict[str, pd.DataFrame]:
    processed_dir = Path(processed_data_dir)
    
    return {
        "match": pd.read_parquet(processed_dir / f"{match_id}_match_info.parquet"),
        "players": pd.read_parquet(processed_dir / f"{match_id}_players_match_info.parquet"),
        "tracking": pd.read_parquet(processed_dir / f"{match_id}_tracking.parquet")
    }