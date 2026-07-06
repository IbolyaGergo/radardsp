#!/usr/bin/bash
FILE_TO_RUN=$1
# Assume environment lives here
ENV="$(pwd)/envs"
CONDA_SCRIPT=$(dirname "$(which conda)")/../etc/profile.d/conda.sh

# Create new window, activate env, and run file in an interactive ipython
# session
tmux new-window -n "ipython-$(basename "$FILE_TO_RUN")" \
    "source $CONDA_SCRIPT && conda activate $ENV && ipython -i $FILE_TO_RUN; exec bash"
