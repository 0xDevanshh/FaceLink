#!/usr/bin/env python3
"""FaceChain — Face ID + Blockchain Verification pipeline (CLI entrypoint).

    python pipeline.py --image samples/subject.jpg

Run `python pipeline.py --help` for every switch.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402

from facechain import PIPELINE_VERSION  # noqa: E402
from facechain.config import settings  # noqa: E402
from facechain.models import Case  # noqa: E402
from facechain.runner import PipelineError, RunOptions, run  # noqa: E402

console = Console(highlight=False)

STAGES = [
    ("input", "Loading & hashing image"),
    ("face", "Detecting + encoding face"),
    ("search", "Reverse image search"),
    ("verify", "Candidate verification"),
    ("evidence", "Evidence bundle + hashes"),
    ("chain", "Blockchain attestation"),
    ("readback", "On-chain read-back verify"),
]
STAGE_INDEX = {name: i for i, (name, _) in enumerate(STAGES)}

TICK, CROSS, DOT = "[green]✓[/green]", "[red]✗[/red]", "[yellow]•[/yellow]"


class Renderer:
    """Prints the staged progress view. Pure presentation, no pipeline logic."""

    def __init__(self, total: int) -> None:
        self.total = total

    def __call__(self, stage: str, status: str, detail: str) -> None:
        root, _, sub = stage.partition(":")
        idx = STAGE_INDEX.get(root)
        label = dict(STAGES).get(root, root)

        if not sub:
            if status == "start":
                console.print(
                    f"[bold cyan][{idx + 1:02d}/{self.total:02d}][/bold cyan] {label}…"
                )
                return
            mark = {"ok": TICK, "fail": CROSS}.get(status, DOT)
            console.print(f"        {mark} {detail}" if detail else f"        {mark}")
            return

        mark = {"ok": TICK, "fail": CROSS, "start": "[dim]…[/dim]"}.get(status, DOT)
        name = sub.replace("_", " ")
        if status == "start":
            console.print(f"        [dim]├─ {name}…[/dim]")
        else:
            console.print(f"        ├─ {name:<22} {mark} [dim]{detail}[/dim]")


def header() -> None:
    console.print(
        Panel(
            Text.from_markup(
                "[bold white]FACECHAIN VERIFICATION PIPELINE[/bold white]\n"
                f"[dim]v{PIPELINE_VERSION} • {settings.chain_name} testnet • EAS attestations[/dim]",
                justify="center",
            ),
            border_style="cyan",
            padding=(0, 4),
        )
    )


def summary(case: Case) -> None:
    verdict_style = {
        "VERIFIED": "bold green",
        "VERIFIED_SIMULATED": "bold yellow",
        "VERIFIED_OFFCHAIN": "bold yellow",
    }.get(case.verdict, "bold red")

    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right")
    table.add_column()

    best = case.best_match
    if best:
        table.add_row("Match score", f"{best.final_score * 100:.1f}%")
        table.add_row("Image similarity", f"{best.image_similarity * 100:.1f}%")
        table.add_row("Face similarity", f"{best.face_similarity * 100:.1f}%")
        table.add_row("Platform", best.platform or "n/a")
        table.add_row("Matched URL", best.url)
        table.add_row("Found via", best.engine)
        table.add_row("Ladder", " → ".join(s.value for s in best.stages))

    chain = case.blockchain
    if chain and chain.mode not in ("skipped",):
        table.add_row("Network", f"{chain.network} ({chain.chain_id})")
        if chain.schema_uid:
            table.add_row("Schema UID", chain.schema_uid)
        if chain.tx_hash:
            table.add_row("Tx hash", chain.tx_hash)
        if chain.attestation_uid:
            table.add_row("Attestation UID", chain.attestation_uid)
        if chain.explorer_attestation:
            table.add_row("EAS explorer", chain.explorer_attestation)
        if chain.mode == "onchain":
            table.add_row(
                "Read-back", "PASS" if chain.readback_verified else "FAIL"
            )
        if chain.note:
            table.add_row("Note", chain.note)

    if case.failure_reason:
        table.add_row("Reason", case.failure_reason)

    console.print()
    console.print(Panel(table, title=f"[{verdict_style}]{case.verdict}[/{verdict_style}]",
                        border_style="cyan", padding=(1, 2)))
    console.print(f"\nEvidence bundle: [bold]{settings.evidence_dir / case.case_id}[/bold]")
    console.print(
        "[dim]Scope: attests that the input image and its primary face match the retrieved "
        "public image under the recorded thresholds. Not an identity claim.[/dim]"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="pipeline.py",
        description="Detect a face, find a real matching social post via reverse image "
                    "search, and attest the match on an EVM testnet via EAS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image", required=True, help="path to the input photo")
    p.add_argument("--image-url", default=None,
                   help="public URL of the same image (enables engines' by-URL search; "
                        "nothing is uploaded anywhere)")
    p.add_argument("--engines", default=settings.engines,
                   help="comma-separated: google_lens,yandex,bing,tineye,"
                        "serpapi_google_lens,serpapi_yandex")
    p.add_argument("--allow-upload-host", action="store_true",
                   help="upload the photo to a temporary public host (Litterbox, 1h TTL) "
                        "so by-URL search works with a local file — off by default")
    p.add_argument("--face-backend", choices=["auto", "insightface", "opencv"], default=None)
    p.add_argument("--max-verify", type=int, default=None,
                   help="max candidates to fetch and measure")
    p.add_argument("--scan-depth", choices=["fast", "standard", "deep"], default="standard",
                   help="fast: 5 candidates, standard: 12, deep: 30 with max discovery")
    p.add_argument("--case-id", default=None, help="override the generated case id")

    chain = p.add_mutually_exclusive_group()
    chain.add_argument("--no-chain", action="store_true",
                      help="stop after local verification (no transaction)")
    chain.add_argument("--simulate", action="store_true",
                      help="eth_call the attestation instead of sending it: validates "
                           "encoding and contract call, costs no gas, needs no funds")

    p.add_argument("--headful", action="store_true", help="show the browser (great for demos)")
    p.add_argument("--json", action="store_true", help="print case.json to stdout at the end")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level={0: logging.WARNING, 1: logging.INFO}.get(args.verbose, logging.DEBUG),
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if args.headful:
        settings.headless = False
    if args.allow_upload_host:
        settings.allow_upload_host = True

    chain_mode = "skip" if args.no_chain else ("simulate" if args.simulate else "onchain")
    opts = RunOptions(
        image=args.image,
        image_url=args.image_url,
        engines=[e.strip() for e in args.engines.split(",") if e.strip()],
        chain_mode=chain_mode,
        face_backend=args.face_backend,
        max_verify=args.max_verify,
        case_id=args.case_id,
        scan_depth=args.scan_depth,
    )

    header()
    console.print(
        f"[dim]input={args.image}  engines={','.join(opts.engines)}  chain={chain_mode}[/dim]\n"
    )

    try:
        case = run(opts, Renderer(len(STAGES)))
    except PipelineError as exc:
        console.print(f"\n{CROSS} [bold red]{exc}[/bold red]")
        return 2
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted[/yellow]")
        return 130

    summary(case)
    if args.json:
        console.print_json(json.dumps(case.model_dump(mode="json")))

    return 0 if case.verdict.startswith("VERIFIED") and case.verdict != "VERIFIED_OFFCHAIN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
