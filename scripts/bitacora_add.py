#!/usr/bin/env python3
import argparse
import datetime as dt
import os
import sys
from typing import List

BITACORA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "BITACORA.md")


def ensure_file_exists(path: str) -> None:
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("## Bitácora del Proyecto DeepLearningP2\n\n")


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()


def write_lines(path: str, lines: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def add_entry(message: str, tag: str | None) -> None:
    today = dt.date.today().isoformat()
    time_now = dt.datetime.now().strftime("%H:%M")
    ensure_file_exists(BITACORA_PATH)
    lines = read_lines(BITACORA_PATH)

    # Find date header
    date_header = f"### {today}\n"
    try:
        idx = lines.index(date_header)
    except ValueError:
        # Append a new date section at the end
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        if lines and lines[-1].strip() != "":
            lines.append("\n")
        lines.append(date_header)
        idx = len(lines) - 1
        lines.append(f"- {time_now} — {message}\n" if not tag else f"- {time_now} — [{tag}] {message}\n")
        write_lines(BITACORA_PATH, lines)
        return

    # Insert after the header (keep entries in chronological order of addition)
    insert_pos = idx + 1
    lines.insert(insert_pos, f"- {time_now} — {message}\n" if not tag else f"- {time_now} — [{tag}] {message}\n")
    write_lines(BITACORA_PATH, lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Añade una entrada a BITACORA.md")
    parser.add_argument(
        "-m",
        "--message",
        required=True,
        help="Mensaje del hito a registrar",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Etiqueta opcional del hito (p.ej., DATA, TRAIN, EVAL)",
    )
    args = parser.parse_args()
    try:
        add_entry(args.message, args.tag)
        print(f"Entrada registrada en {BITACORA_PATH}")
        return 0
    except Exception as e:
        print(f"Error al registrar entrada: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


