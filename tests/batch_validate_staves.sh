#!/bin/bash
# If desired files are in "$DIR":
# (for f in $(echo $DIR/IMSLP*.pdf | sort -R | tail -5000); do basename "$f" > /dev/tty; echo "$f"; done) | bash tests/batch_validate_staves.sh

process_file() {
    python tests/validate_staves.py "$1" "tests/validated/$(basename "$1" .pdf).csv.gz"
}

export -f process_file

xargs -n 1 -P 4 bash -c 'process_file "$@"' --
