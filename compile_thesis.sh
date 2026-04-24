#!/usr/bin/env bash

typst compile paper/main.typ

pdftk paper/assignment.pdf paper/main.pdf cat output paper/full_thesis.pdf
