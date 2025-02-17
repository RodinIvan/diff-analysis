#!/bin/bash

set -e

docker build -t txtarcan-interview:snapshot .

docker run -it --rm --name txtarcan \
	--volume ./data:/data \
	--volume ./main.py:/app/main.py \
	txtarcan-interview:snapshot \
	$@
