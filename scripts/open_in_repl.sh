#!/usr/bin/bash
FILE_TO_RUN=$1

# Create new window, activate env, and run file in an interactive ipython
# session
tmux new-window -n "ipython-$(basename "$FILE_TO_RUN")" \
    "bash -ic 'ipython -i \"$FILE_TO_RUN\"; exec bash'"
