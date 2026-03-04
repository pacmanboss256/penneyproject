from __future__ import annotations

from pathlib import Path
import os
import time
import sys
import subprocess
import platform
import shutil
from importlib.machinery import EXTENSION_SUFFIXES

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

FIGURES_DIR = BASE_DIR / "figures"
DATA_DIR = BASE_DIR / "data"


def _list_saved_deck_dirs() -> list[Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return sorted([p for p in DATA_DIR.iterdir() if p.is_dir() and p.name.endswith("_decks")])


def _folder_to_save_name(folder_name: str) -> str:
    if folder_name.endswith("_decks"):
        return folder_name[: -len("_decks")]
    return folder_name


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
                    yield Select([("3", "3"), ("4", "4")], id="bits", value="4")
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
        if not deck_value or deck_value == "__new__":
            save_name = f"deck-{int(time.time())}"
            deck_folder_name = f"{save_name}_decks"
            deck_folder = DATA_DIR / deck_folder_name
        else:
            deck_folder_name = deck_value
            save_name = _folder_to_save_name(deck_folder_name)
            deck_folder = DATA_DIR / deck_folder_name

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        existing_decks: list[str] = []
        if deck_folder.exists():
            self.call_from_thread(self._set_status, f"Loading decks from {deck_folder.name}...")
            existing_decks = saving.load_decks(str(deck_folder))._decks
        else:
            self.call_from_thread(self._set_status, f"Creating new deck set {deck_folder_name}...")

        total = additional
        generated = 0
        if additional >= 100000:
            self.call_from_thread(self._set_progress, 0, total)
        else:
            self.call_from_thread(self._set_progress, 0, 0)

        if existing_decks:
            self.call_from_thread(self._set_status, "Scoring existing decks...")
            base_decks = Deck(existing_decks)
            parser_tricks = Parser(base_decks, bits=bits, scoring_by_tricks=True)
            parser_tricks.raw_out()
            parser_cards = Parser(base_decks, bits=bits, scoring_by_tricks=False)
            parser_cards.raw_out()
        else:
            first_chunk = min(additional, 10000 if additional >= 100000 else additional)
            self.call_from_thread(self._set_status, f"Generating {first_chunk} initial decks...")
            seed_decks = deck_gen(num_decks=first_chunk)
            saving.save_decks(seed_decks, filename=save_name)
            generated += first_chunk
            remaining = additional - first_chunk
            base_decks = Deck(seed_decks._decks)
            parser_tricks = Parser(base_decks, bits=bits, scoring_by_tricks=True)
            parser_tricks.raw_out()
            parser_cards = Parser(base_decks, bits=bits, scoring_by_tricks=False)
            parser_cards.raw_out()
            if additional >= 100000:
                self.call_from_thread(self._set_progress, generated, total)
        if existing_decks:
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
                    parser_tricks.add_decks(chunk, decks=new_decks)
                    parser_cards.add_decks(chunk, decks=new_decks)
                    saving.save_decks(new_decks, filename=save_name)
                    existing_decks.extend(new_decks._decks)
                    generated += chunk
                    self.call_from_thread(self._set_progress, generated, total)
            else:
                self.call_from_thread(self._set_status, f"Generating {remaining} decks...")
                new_decks = deck_gen(num_decks=remaining)
                parser_tricks.add_decks(remaining, decks=new_decks)
                parser_cards.add_decks(remaining, decks=new_decks)
                saving.save_decks(new_decks, filename=save_name)
                existing_decks.extend(new_decks._decks)
                generated += remaining
        else:
            existing_decks = base_decks._decks

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
        self.call_from_thread(self._set_status, f"Re-scoring {len(decks)} decks by {method}...")
        parser = Parser(Deck(decks), bits=bits, scoring_by_tricks=(method == "tricks"))
        parser.raw_out()
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
