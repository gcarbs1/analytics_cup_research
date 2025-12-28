"""
get_data.py
Author: Gabriel Carbinatto (gabrielcarbinatto@usp.br)
"""

import subprocess
import json
import platform
import shutil
import stat
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple


class SkillCornerCollector:
    REPO_URL = "https://github.com/SkillCorner/opendata.git"

    def __init__(
        self,
        repo_dir: str = "opendata_repo",
        data_root: str = "data/raw",
    ):
        self.repo_dir = Path(repo_dir)

        self.data_root = Path(data_root)
        self.match_out = self.data_root / "match"
        self.tracking_out = self.data_root / "tracking"
        self.dynamic_events_out = self.data_root / "dynamic_events"
        self.phases_out = self.data_root / "phases_of_play"

        self.match_out.mkdir(parents=True, exist_ok=True)
        self.tracking_out.mkdir(parents=True, exist_ok=True)
        self.dynamic_events_out.mkdir(parents=True, exist_ok=True)
        self.phases_out.mkdir(parents=True, exist_ok=True)

    def _run(self, cmd: List[str], cwd: Path = None, check: bool = True) -> str:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        if check and result.returncode != 0:
            raise RuntimeError(
                f"Command failed: {' '.join(cmd)}\n{result.stderr or result.stdout}"
            )
        return result.stdout.strip()

    def _check(self, cmd: List[str]) -> bool:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _get_linux_distro(self) -> str:
        try:
            return Path("/etc/os-release").read_text(encoding="utf-8").lower()
        except FileNotFoundError:
            return ""

    def install_git_lfs(self) -> None:
        system = platform.system().lower()

        if "windows" in system:
            raise RuntimeError(
                "git-lfs not found. Please install manually on Windows: https://git-lfs.com/ "
                "Then run again."
            )

        if "darwin" in system:
            if not self._check(["brew", "--version"]):
                raise RuntimeError(
                    "Homebrew not found. Install Homebrew (https://brew.sh) "
                    "and then run: brew install git-lfs"
                )
            self._run(["brew", "install", "git-lfs"])

        elif "linux" in system:
            distro = self._get_linux_distro()
            if "ubuntu" in distro or "debian" in distro:
                self._run(["sudo", "apt-get", "update"])
                self._run(["sudo", "apt-get", "install", "-y", "git-lfs"])
            elif "fedora" in distro or "rhel" in distro or "centos" in distro:
                self._run(["sudo", "dnf", "install", "-y", "git-lfs"])
            elif "arch" in distro:
                self._run(["sudo", "pacman", "-S", "--noconfirm", "git-lfs"])
            else:
                raise RuntimeError(
                    "Linux distribution not automatically supported. "
                    "Install git-lfs manually (https://git-lfs.com/)."
                )
        else:
            raise RuntimeError(
                "Operating system not automatically supported for git-lfs installation."
            )

        self._run(["git", "lfs", "install"], check=False)

    def setup_dependencies(self) -> None:
        if not self._check(["git", "--version"]):
            raise RuntimeError("git not found. Install git and run again.")

        if not self._check(["git", "lfs", "version"]):
            self.install_git_lfs()
            if not self._check(["git", "lfs", "version"]):
                raise RuntimeError(
                    "git-lfs still not available after installation attempt. "
                    "Install manually and run again."
                )

        self._run(["git", "lfs", "install"], check=False)

    def clone_repository(self) -> None:
        data_dir = self.repo_dir / "data"

        if self.repo_dir.exists() and not data_dir.exists():
            print(f"Incomplete repository detected at {self.repo_dir}. Removing...")
            shutil.rmtree(self.repo_dir, ignore_errors=True)

        if not self.repo_dir.exists():
            print(f"Cloning repository to {self.repo_dir}...")
            self._run(["git", "clone", self.REPO_URL, str(self.repo_dir)])
        else:
            print(f"Repository already exists at {self.repo_dir}")

        print("Setting up git-lfs...")
        self._run(["git", "lfs", "install"], cwd=self.repo_dir, check=False)
        print("Pulling LFS files (this may take a while)...")
        self._run(["git", "lfs", "pull"], cwd=self.repo_dir)
        print("LFS pull completed.")

    def get_match_ids(self) -> List[str]:
        matches_dir = self.repo_dir / "data" / "matches"
        if not matches_dir.exists():
            print(f"Expected directory not found: {matches_dir}")
            print(f"Repository root exists: {self.repo_dir.exists()}")
            if self.repo_dir.exists():
                print(f"Contents of {self.repo_dir}:")
                for item in self.repo_dir.iterdir():
                    print(f"  - {item.name}")
            raise FileNotFoundError(f"Directory not found: {matches_dir}")

        match_ids: List[str] = []

        for item in matches_dir.iterdir():
            if item.is_dir() and item.name.isdigit():
                mid = item.name
                match_json_path = item / f"{mid}_match.json"
                tracking_path = item / f"{mid}_tracking_extrapolated.jsonl"

                if not match_json_path.exists() or not tracking_path.exists():
                    continue

                with open(tracking_path, "rb") as f:
                    first_bytes = f.read(100)
                if first_bytes.startswith(b"version https://git-lfs"):
                    continue

                match_ids.append(mid)

        match_ids.sort(key=int)
        return match_ids

    def _copy_flat_files(self, match_id: str) -> Tuple[Path, Path, Path, Path]:
        src_dir = self.repo_dir / "data" / "matches" / match_id

        src_match = src_dir / f"{match_id}_match.json"
        src_track = src_dir / f"{match_id}_tracking_extrapolated.jsonl"
        src_dynamic = src_dir / f"{match_id}_dynamic_events.csv"
        src_phases = src_dir / f"{match_id}_phases_of_play.csv"

        dst_match = self.match_out / f"{match_id}_match.json"
        dst_track = self.tracking_out / f"{match_id}_tracking_extrapolated.jsonl"
        dst_dynamic = self.dynamic_events_out / f"{match_id}_dynamic_events.csv"
        dst_phases = self.phases_out / f"{match_id}_phases_of_play.csv"

        shutil.copy2(src_match, dst_match)
        shutil.copy2(src_track, dst_track)

        if src_dynamic.exists():
            shutil.copy2(src_dynamic, dst_dynamic)
        if src_phases.exists():
            shutil.copy2(src_phases, dst_phases)

        return dst_match, dst_track, dst_dynamic, dst_phases

    def prepare_data(self) -> List[Dict[str, Any]]:
        self.setup_dependencies()
        self.clone_repository()

        match_ids = self.get_match_ids()

        summary: List[Dict[str, Any]] = []

        for mid in match_ids:
            src_match_file = (
                self.repo_dir / "data" / "matches" / mid / f"{mid}_match.json"
            )
            src_tracking_file = (
                self.repo_dir
                / "data"
                / "matches"
                / mid
                / f"{mid}_tracking_extrapolated.jsonl"
            )

            with open(src_match_file, "r", encoding="utf-8") as f:
                match_json = json.load(f)

            home_team = match_json.get("home_team", {}).get("name", None)
            away_team = match_json.get("away_team", {}).get("name", None)

            dst_match_file, dst_tracking_file, dst_dynamic_file, dst_phases_file = self._copy_flat_files(mid)

            frame_count = 0
            with open(src_tracking_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        frame_count += 1

            summary.append(
                {
                    "match_id": mid,
                    "home_team": home_team,
                    "away_team": away_team,
                    "tracking_frames": frame_count,
                    "match_file": str(dst_match_file),
                    "tracking_file": str(dst_tracking_file),
                    "dynamic_events_file": str(dst_dynamic_file),
                    "phases_of_play_file": str(dst_phases_file),
                }
            )

        summary_path = self.data_root / "matches_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"Removing repository directory: {self.repo_dir}")
        if self.repo_dir.exists():
            def remove_readonly(func, path, excinfo):
                os.chmod(path, stat.S_IWRITE)
                func(path)
            shutil.rmtree(self.repo_dir, onerror=remove_readonly)
            print(f"Repository directory removed successfully.")

        return summary