#!/bin/bash

$ -cwd

. /etc/profile.d/modules.sh
module load python/3.11.4

# Run the executable
pipenv install
pipenv shell
