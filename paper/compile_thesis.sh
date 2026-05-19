#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-assignment_first}"

typst compile main.typ main.pdf

if [[ "$MODE" == "title_first" ]]; then  # Title page first:
    # main pages 1-2, then assignment, then rest of main
    pdftk \
        A=main.pdf \
        B=assignment.pdf \
        cat A1-2 B1-end A3-end \
        output full_thesis.pdf

else  # Default: assignment page first
    pdftk assignment.pdf main.pdf cat output full_thesis.pdf
fi

gs  -o  final_PRINT.pdf  -dNoOutputFonts  -sDEVICE=pdfwrite  full_thesis.pdf
