import argparse
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

@dataclass
class Attempt:
    puzzle_id: str
    puzzle_type: str
    red_pixels: Optional[int]
    flags: dict[str, bool]  # overlaps, start, goal, connected
    message: str
    
    @property
    def success(self) -> bool:
        # Success if overlaps=False, others=True
        return (self.flags.get('overlaps_walls') is False and
                self.flags.get('touches_start') is True and
                self.flags.get('touches_goal') is True and
                self.flags.get('connected') is True)

def safe_cast(val: Any, cast_type: type, default: Any = None) -> Any:
    """Removes the 50 lines of coercion logic from original."""
    try:
        if cast_type is bool and isinstance(val, str):
            return val.lower() in ('true', '1', 'yes')
        return cast_type(val)
    except (ValueError, TypeError):
        return default

def parse_evaluation(path: Path) -> Optional[Attempt]:
    try:
        payload = json.loads(path.read_text())
        # Handle the double-encoded JSON in stdout
        raw_stdout = payload.get("stdout", "")
        if not isinstance(raw_stdout, str) or not raw_stdout.strip():
            return None
            
        data = json.loads(raw_stdout)
        
        return Attempt(
            puzzle_id=str(data.get("puzzle_id", "unknown")),
            puzzle_type=path.parent.name.split("_")[0],
            red_pixels=safe_cast(data.get("red_pixel_count"), int),
            flags={
                'overlaps_walls': safe_cast(data.get("overlaps_walls"), bool),
                'touches_start': safe_cast(data.get("touches_start"), bool),
                'touches_goal': safe_cast(data.get("touches_goal"), bool),
                'connected': safe_cast(data.get("connected"), bool),
            },
            message=str(data.get("message", ""))
        )
    except Exception:
        return None

def summarize(roots: list[Path], top_fail: int):
    attempts = [res for root in roots for p in root.rglob("evaluation.json") 
                if (res := parse_evaluation(p))]
    if not attempts:
        print("No evaluations found.")
        return

    # --- Statistics ---
    successes = [a for a in attempts if a.success]
    print(f"Total: {len(attempts)} | Success: {len(successes)} ({len(successes)/len(attempts):.1%})\n")

    # Grouping
    by_type = defaultdict(list)
    for a in attempts: by_type[a.puzzle_type].append(a)

    print("--- By Type ---")
    for p_type, records in sorted(by_type.items()):
        s_rate = sum(1 for r in records if r.success) / len(records)
        print(f"{p_type:<15} {len(records):>4} runs | {s_rate:>6.1%} success")

    print("\n--- Diagnostics (Failures) ---")
    print(f"Wall Collisions : {sum(1 for a in attempts if a.flags['overlaps_walls'])}")
    print(f"Missed Start    : {sum(1 for a in attempts if not a.flags['touches_start'])}")
    print(f"Missed Goal     : {sum(1 for a in attempts if not a.flags['touches_goal'])}")
    print(f"Disconnected    : {sum(1 for a in attempts if not a.flags['connected'])}")

    pixels = [a.red_pixels for a in attempts if a.red_pixels is not None]
    if pixels:
        print(f"\nAvg Red Pixels  : {statistics.mean(pixels):.1f}")

    if top_fail > 0:
        print(f"\n--- Top {top_fail} Difficult Puzzles ---")
        # Map puzzle_id -> list of attempts
        by_id = defaultdict(list)
        for a in attempts: by_id[f"{a.puzzle_type}/{a.puzzle_id}"].append(a)
        
        # Calculate failure rate
        stats = []
        for pid, recs in by_id.items():
            wins = sum(1 for r in recs if r.success)
            if wins < len(recs): # Only show if there was a failure
                stats.append((wins/len(recs), pid, len(recs)))
        
        for rate, pid, count in sorted(stats)[:top_fail]:
            print(f"{pid:<30} Success: {rate:.0%} ({count} attempts)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, action="append", default=[Path("data/output")])
    parser.add_argument("--top-failures", type=int, default=5)
    args = parser.parse_args()
    summarize(args.root, args.top_failures)

if __name__ == "__main__":
    main()