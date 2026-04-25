#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-assignment_first}"

typst compile paper/main.typ paper/main.pdf

if [[ "$MODE" == "title_first" ]]; then  # Title page first:
    # main pages 1-2, then assignment, then rest of main
    pdftk \
        A=paper/main.pdf \
        B=paper/assignment.pdf \
        cat A1-2 B1-end A3-end \
        output paper/full_thesis.pdf

else  # Default: assignment page first
    pdftk paper/assignment.pdf paper/main.pdf cat output paper/full_thesis.pdf
fi
