from __future__ import annotations

from pathlib import Path
import os
import json
import time
import sys
import subprocess
import platform
import shutil
from concurrent.futures import ThreadPoolExecutor
from importlib.machinery import EXTENSION_SUFFIXES
from itertools import permutations, repeat

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Footer, Header, Input, Label, Static, TabbedContent, TabPane, Select, ProgressBar
from textual.worker import get_current_worker

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

# compile the cython speedup modules before anything else!!!


def _latest_mtime(paths: list[Path]) -> float:
    latest = 0.0
    for path in paths:
        if path.exists():
            latest = max(latest, path.stat().st_mtime)
    return latest


def _cython_built() -> bool:
    machine = platform.machine().lower()
    is_x86 = machine in {"x86_64", "amd64", "i386", "i686"}
    expected = ("parser", "fastmatch", "fastmatch_simd") if is_x86 else ("parser", "fastmatch")
    for name in expected:
        built_targets = [SRC_DIR / f"{name}{suffix}" for suffix in EXTENSION_SUFFIXES]
        built_path = next((target for target in built_targets if target.exists()), None)
        if built_path is None:
            return False
        source_paths = [SRC_DIR / f"{name}.pyx", SRC_DIR / f"{name}.pxd", SRC_DIR / f"{name}.py"]
        if built_path.stat().st_mtime < _latest_mtime(source_paths):
            return False
    return True


# this function here is written by codex because i dont want to deal with macOS

def _ensure_macos_prereqs() -> None:
    if sys.platform != "darwin":
        return

    def _xcrun_ready() -> bool:
        if shutil.which("xcrun") is None:
            return False
        probe = subprocess.run(["xcrun", "--version"], capture_output=True, text=True)
        return probe.returncode == 0

    if _xcrun_ready():
        return

    # start the installer UI for CLT (if not already running)
    subprocess.run(["xcode-select", "--install"], check=False, capture_output=True, text=True)

    clt_path = Path("/Library/Developer/CommandLineTools")
    if clt_path.exists():
        # try to auto-fix invalid active developer path after CLT appears on disk
        subprocess.run(["xcode-select", "--switch", str(clt_path)], check=False, capture_output=True, text=True)
        subprocess.run(["xcode-select", "--reset"], check=False, capture_output=True, text=True)

    # wait for installation to complete and become usable
    for _ in range(60):
        if _xcrun_ready():
            return
        time.sleep(5)

    raise RuntimeError(
        "Started macOS Command Line Tools installation. Complete the installer dialog, "
        "then rerun `uv run main.py`."
    )


def _ensure_cython_built() -> None:
    if _cython_built():
        return
    _ensure_macos_prereqs()
    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=str(SRC_DIR),
        check=True,
    )


_ensure_cython_built()

from src.decks import Deck, deck_gen
from src import saving
from src.parser import Parser
from src.heatmaps import make_heatmap
from src.scores import ScoreTable, load_table
try:
    from src.fastmatch_simd import winner_counts_for_pair
except Exception:
    from src.fastmatch import winner_counts_for_pair

FIGURES_DIR = BASE_DIR / "figures"
DATA_DIR = BASE_DIR / "data"

_SCORE_CACHE_VERSION = 2


def _score_cache_tag(by_tricks: bool) -> str:
    return "tricks" if by_tricks else "cards"


def _score_cache_csv_path(deck_folder: Path, bits: int, by_tricks: bool) -> Path:
    return deck_folder / f"scores_bits{bits}_{_score_cache_tag(by_tricks)}.csv"


def _score_cache_meta_path(deck_folder: Path, bits: int, by_tricks: bool) -> Path:
    return deck_folder / f"scores_bits{bits}_{_score_cache_tag(by_tricks)}.meta.json"


def _pair_options(bits: int) -> list[tuple[str, str]]:
    player_options = [str(bin(w))[2:].zfill(bits) for w in range(2**bits)]
    return list(permutations(player_options, 2))


def _score_workers(pair_count: int) -> int:
    return max(1, min(pair_count, os.cpu_count() or 1))


def _score_pair_rows(
    decks_bytes: list[bytes], score_by_tricks: bool, pair: tuple[str, str]
) -> list[int | str]:
    p1, p2 = pair
    counts = winner_counts_for_pair(decks_bytes, p1, p2, aligned=False, score_by_tricks=score_by_tricks)
    return [p1, p2, int(counts[0]), int(counts[1]), int(counts[2])]


def _score_rows_parallel(decks: list[str], bits: int, score_by_tricks: bool) -> list[list[int | str]]:
    decks_bytes = [deck.encode("ascii") for deck in decks]
    pairs = _pair_options(bits)
    workers = _score_workers(len(pairs))
    if workers == 1:
        return [_score_pair_rows(decks_bytes, score_by_tricks, pair) for pair in pairs]

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="penney-score") as executor:
        return list(executor.map(_score_pair_rows, repeat(decks_bytes), repeat(score_by_tricks), pairs))


def _merge_score_rows(current_scores: list, additional_scores: list) -> list:
    if not current_scores:
        return [list(row) for row in additional_scores]

    merged_scores = []
    for current_row, add_row in zip(current_scores, additional_scores):
        if current_row[0] != add_row[0] or current_row[1] != add_row[1]:
            raise ValueError("Score rows are misaligned and cannot be merged.")
        merged_scores.append(
            [
                current_row[0],
                current_row[1],
                int(current_row[2]) + int(add_row[2]),
                int(current_row[3]) + int(add_row[3]),
                int(current_row[4]) + int(add_row[4]),
            ]
        )
    return merged_scores


def _load_score_cache(deck_folder: Path, bits: int, by_tricks: bool, deck_count: int) -> list | None:
    csv_path = _score_cache_csv_path(deck_folder, bits, by_tricks)
    meta_path = _score_cache_meta_path(deck_folder, bits, by_tricks)
    if not csv_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return None

    if (
        meta.get("version") != _SCORE_CACHE_VERSION
        or meta.get("bits") != bits
        or meta.get("method") != _score_cache_tag(by_tricks)
        or meta.get("deck_count") != deck_count
    ):
        return None

    try:
        table = load_table(csv_path, scoring_by_tricks=by_tricks)
    except Exception:
        return None
    return table.raw.values.tolist()


def _save_score_cache(deck_folder: Path, bits: int, by_tricks: bool, scores: list, deck_count: int) -> None:
    deck_folder.mkdir(parents=True, exist_ok=True)
    csv_path = _score_cache_csv_path(deck_folder, bits, by_tricks)
    meta_path = _score_cache_meta_path(deck_folder, bits, by_tricks)

    ScoreTable(scores, scoring_by_tricks=by_tricks).save(csv_path)
    meta = {
        "version": _SCORE_CACHE_VERSION,
        "bits": bits,
        "method": _score_cache_tag(by_tricks),
        "deck_count": deck_count,
        "saved_at": int(time.time()),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def _list_saved_deck_dirs() -> list[Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    deck_dirs = []
    for path in DATA_DIR.iterdir():
        if not path.is_dir():
            continue
        if (path / "metadata.json").exists():
            deck_dirs.append(path)
    return sorted(deck_dirs)


def _latest_heatmaps() -> list[Path]:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    files = [p for p in FIGURES_DIR.iterdir() if p.is_file() and "heatmap" in p.name]
    if not files:
        return []

    def _newest_with_tag(tag: str) -> Path | None:
        tagged = [p for p in files if tag in p.name]
        if not tagged:
            return None
        # mstat ordering lets us pick the most recently modified file
        return max(tagged, key=lambda p: p.stat().st_mtime)

    picks: list[Path] = []
    for tag in ("tricks", "cards"):
        pick = _newest_with_tag(tag)
        if pick:
            picks.append(pick)

    if picks:
        return picks
    # mstat ordering lets us pick the most recently modified file
    return [max(files, key=lambda p: p.stat().st_mtime)]


def _open_heatmaps_window(paths: list[Path]) -> None:
    script = (
        "import sys\n"
        "import matplotlib.pyplot as plt\n"
        "paths = sys.argv[1:]\n"
        "cols = len(paths)\n"
        # bigger figsize slows down rendering a lot
        "fig, axes = plt.subplots(1, cols, figsize=(9 * cols, 9))\n"
        "if cols == 1:\n"
        "    axes = [axes]\n"
        "for ax, path in zip(axes, paths):\n"
        "    img = plt.imread(path)\n"
        "    ax.imshow(img)\n"
        "    ax.set_title(path.split('/')[-1])\n"
        "    ax.axis('off')\n"
        "fig.tight_layout()\n"
        "plt.show()\n"
    )
    # open in subprocesses so that the TUI can be changed in main thread while heatmap is open
    subprocess.Popen([sys.executable, "-c", script, *[str(p) for p in paths]])


class PenneyApp(App):
    CSS_PATH = "penneyapp.tcss"

    BINDINGS = [("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(id="tabs"):
            with TabPane("View Latest Heatmaps", id="tab-view"):
                yield Button("View Latest Heatmaps", id="view")
            with TabPane("Generate Additional Decks", id="tab-generate"):
                with Horizontal(id="input-row"):
                    yield Label("Base deck file:")
                    yield Select([], id="deck-file")
                    yield Label("Additional decks:")
                    yield Input(placeholder="e.g. 10000", id="deck-count", restrict=r"[0-9]*")
                    yield Button("Run Update", id="run")
            with TabPane("Bit Selection", id="tab-bits"):
                with Horizontal(id="bits-row"):
                    yield Label("Bits:")
                    yield Select([("3", "3"), ("4", "4")], id="bits", value="3")
            with TabPane("Scoring Method", id="tab-score"):
                with Horizontal(id="score-row"):
                    yield Label("Deck file:")
                    yield Select([], id="score-deck-file")
                    yield Label("Scoring:")
                    yield Select([("By Tricks", "tricks"), ("By Cards", "cards")], id="score-method", value="tricks")
                    yield Button("Re-score", id="rescore")
        yield Static("", id="status")
        with Horizontal(id="progress-row"):
            yield Label("", id="progress-label")
            yield ProgressBar(total=1, id="progress")
        yield Static("", id="output")
        yield Footer()

    def _set_status(self, message: str) -> None:
        self.query_one("#status", Static).update(message)

    # progress bar for when generating >= 100000 additional decks
    def _set_progress(self, current: int, total: int) -> None:
        label = self.query_one("#progress-label", Label)
        bar = self.query_one("#progress", ProgressBar)
        if total <= 0:
            label.update("")
            bar.update(total=1, progress=0)
            bar.styles.display = "none"
            return
        label.update(f"Generated {current} / {total} decks")
        bar.update(total=total, progress=current)
        bar.styles.display = "block"

    # lets us select the deck files
    def _refresh_deck_file_options(self) -> None:
        deck_dirs = _list_saved_deck_dirs()
        options = [("Create new deck set", "__new__")] + [(p.name, p.name) for p in deck_dirs]
        deck_select = self.query_one("#deck-file", Select)
        deck_select.set_options(options)
        valid_values = {v for v, _ in options}
        if deck_dirs:
            if deck_select.value not in valid_values or deck_select.value == "__new__":
                deck_select.value = deck_dirs[0].name
        else:
            deck_select.value = "__new__"

        score_options = [(p.name, p.name) for p in deck_dirs]
        score_select = self.query_one("#score-deck-file", Select)
        if score_options:
            score_select.set_options(score_options)
            if score_select.value not in {v for v, _ in score_options}:
                score_select.value = score_options[0][0]
        else:
            score_select.set_options([("No deck files found", "")])
            score_select.value = ""

    def _run_update(self) -> None:
        raw = self.query_one("#deck-count", Input).value.strip()
        bits_raw = self.query_one("#bits", Select).value
        deck_value = self.query_one("#deck-file", Select).value
        if not raw.isdigit():
            self._set_status("Enter a positive integer for additional decks.")
            return
        additional = int(raw)
        if additional <= 0:
            self._set_status("Additional decks must be > 0.")
            return
        bits = int(bits_raw)
        self._set_status("Starting update...")
        self.run_worker(
            lambda: self._update_data_and_figures(additional, bits, deck_value), thread=True, exclusive=True
        )

    def _show_heatmaps(self) -> None:
        paths = _latest_heatmaps()
        if not paths:
            self._set_status("No heatmaps found in figures/.")
            self.query_one("#output", Static).update("")
            return
        self._set_status("Opening heatmaps in a new window.")
        self.query_one("#output", Static).update("")
        _open_heatmaps_window(paths)

    def _update_data_and_figures(self, additional: int, bits: int, deck_value: str) -> None:
        worker = get_current_worker()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        scoring_workers = _score_workers(len(_pair_options(bits)))
        if not deck_value or deck_value == "__new__":
            deck_folder_name = f"deck-{int(time.time())}_decks"
            deck_folder = DATA_DIR / deck_folder_name
        else:
            deck_folder_name = deck_value
            deck_folder = DATA_DIR / deck_folder_name

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        existing_decks: list[str] = []
        tricks_scores: list = []
        cards_scores: list = []
        if deck_folder.exists():
            self.call_from_thread(self._set_status, f"Loading decks from {deck_folder.name}...")
            existing_decks = saving.load_decks(str(deck_folder))._decks
        else:
            self.call_from_thread(self._set_status, f"Creating new deck set {deck_folder_name}...")
        had_existing_decks = bool(existing_decks)

        total = additional
        generated = 0
        if additional >= 100000:
            self.call_from_thread(self._set_progress, 0, total)
        else:
            self.call_from_thread(self._set_progress, 0, 0)

        if existing_decks:
            cached_tricks = _load_score_cache(deck_folder, bits, True, len(existing_decks))
            if cached_tricks is not None:
                self.call_from_thread(self._set_status, "Loaded cached trick scores.")
                tricks_scores = cached_tricks
            else:
                self.call_from_thread(
                    self._set_status, f"Scoring existing decks (tricks) across {scoring_workers} CPU cores..."
                )
                tricks_scores = _score_rows_parallel(existing_decks, bits, True)

            cached_cards = _load_score_cache(deck_folder, bits, False, len(existing_decks))
            if cached_cards is not None:
                self.call_from_thread(self._set_status, "Loaded cached card scores.")
                cards_scores = cached_cards
            else:
                self.call_from_thread(
                    self._set_status, f"Scoring existing decks (cards) across {scoring_workers} CPU cores..."
                )
                cards_scores = _score_rows_parallel(existing_decks, bits, False)
        else:
            first_chunk = min(additional, 10000 if additional >= 100000 else additional)
            self.call_from_thread(self._set_status, f"Generating {first_chunk} initial decks...")
            seed_decks = deck_gen(num_decks=first_chunk)
            saving.save_decks(seed_decks, filename=deck_folder_name)
            generated += first_chunk
            remaining = additional - first_chunk
            existing_decks = list(seed_decks._decks)
            self.call_from_thread(
                self._set_status, f"Scoring initial decks (tricks) across {scoring_workers} CPU cores..."
            )
            tricks_scores = _score_rows_parallel(existing_decks, bits, True)
            self.call_from_thread(
                self._set_status, f"Scoring initial decks (cards) across {scoring_workers} CPU cores..."
            )
            cards_scores = _score_rows_parallel(existing_decks, bits, False)
            if additional >= 100000:
                self.call_from_thread(self._set_progress, generated, total)
        if had_existing_decks:
            remaining = additional
        if remaining <= 0:
            remaining = 0

        if remaining > 0:
            if additional >= 100000:
                chunk_size = 10000
                while generated < total:
                    if worker.is_cancelled:
                        return
                    chunk = min(chunk_size, total - generated)
                    self.call_from_thread(self._set_status, f"Generating decks {generated + 1}-{generated + chunk}...")
                    new_decks = deck_gen(num_decks=chunk)
                    new_deck_list = list(new_decks._decks)
                    tricks_scores = _merge_score_rows(tricks_scores, _score_rows_parallel(new_deck_list, bits, True))
                    cards_scores = _merge_score_rows(cards_scores, _score_rows_parallel(new_deck_list, bits, False))
                    saving.save_decks(new_decks, filename=deck_folder_name)
                    existing_decks.extend(new_deck_list)
                    generated += chunk
                    self.call_from_thread(self._set_progress, generated, total)
            else:
                self.call_from_thread(self._set_status, f"Generating {remaining} decks...")
                new_decks = deck_gen(num_decks=remaining)
                new_deck_list = list(new_decks._decks)
                tricks_scores = _merge_score_rows(tricks_scores, _score_rows_parallel(new_deck_list, bits, True))
                cards_scores = _merge_score_rows(cards_scores, _score_rows_parallel(new_deck_list, bits, False))
                saving.save_decks(new_decks, filename=deck_folder_name)
                existing_decks.extend(new_deck_list)
                generated += remaining

        parser_tricks = Parser(Deck(existing_decks), bits=bits, scoring_by_tricks=True)
        parser_tricks.scores = tricks_scores
        parser_cards = Parser(Deck(existing_decks), bits=bits, scoring_by_tricks=False)
        parser_cards.scores = cards_scores

        # Persist score caches so subsequent runs can avoid rescoring existing decks.
        try:
            _save_score_cache(deck_folder, bits, True, parser_tricks.scores, len(parser_tricks.decks._decks))
            _save_score_cache(deck_folder, bits, False, parser_cards.scores, len(parser_cards.decks._decks))
        except Exception:
            # Cache write failure shouldn't block figure generation.
            pass

        make_heatmap(parser_tricks.scores, by_tricks=True, parser=parser_tricks)
        make_heatmap(parser_cards.scores, by_tricks=False, parser=parser_cards)

        self.call_from_thread(
            self._set_status, f"Generated {additional} decks in {deck_folder_name} and updated figures."
        )
        self.call_from_thread(self._set_progress, 0, 0)
        self.call_from_thread(self._refresh_deck_file_options)
        self.call_from_thread(self._show_heatmaps)

    def _run_rescore(self) -> None:
        bits_raw = self.query_one("#bits", Select).value
        method = self.query_one("#score-method", Select).value
        deck_value = self.query_one("#score-deck-file", Select).value
        if not deck_value:
            self._set_status("No deck files available to re-score.")
            return
        bits = int(bits_raw)
        self._set_status("Starting re-score...")
        self.run_worker(lambda: self._rescore_existing_decks(bits, method, deck_value), thread=True, exclusive=True)

    def _rescore_existing_decks(self, bits: int, method: str, deck_value: str) -> None:
        deck_folder = DATA_DIR / deck_value
        if not deck_folder.exists():
            self.call_from_thread(self._set_status, f"Deck file {deck_value} not found.")
            return
        self.call_from_thread(self._set_status, f"Loading decks from {deck_folder.name}...")
        decks = saving.load_decks(str(deck_folder))._decks
        if not decks:
            self.call_from_thread(self._set_status, f"No decks found in {deck_folder.name}.")
            return
        scoring_workers = _score_workers(len(_pair_options(bits)))
        self.call_from_thread(
            self._set_status, f"Re-scoring {len(decks)} decks by {method} across {scoring_workers} CPU cores..."
        )
        parser = Parser(Deck(decks), bits=bits, scoring_by_tricks=(method == "tricks"))
        parser.scores = _score_rows_parallel(decks, bits, method == "tricks")
        try:
            _save_score_cache(deck_folder, bits, method == "tricks", parser.scores, len(decks))
        except Exception:
            pass
        make_heatmap(parser.scores, by_tricks=(method == "tricks"), parser=parser)
        self.call_from_thread(self._set_status, f"Re-scored {len(decks)} decks by {method}.")
        self.call_from_thread(self._show_heatmaps)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "view":
            self._show_heatmaps()
            return

        if event.button.id == "run":
            self._run_update()
            return
        if event.button.id == "rescore":
            self._run_rescore()
            return

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        self.query_one("#status", Static).update("")
        self.query_one("#output", Static).update("")
        self._set_progress(0, 0)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "deck-count":
            self._run_update()

    def on_mount(self) -> None:
        self._set_progress(0, 0)
        self._refresh_deck_file_options()


def main() -> None:
    PenneyApp().run()


if __name__ == "__main__":
    main()
