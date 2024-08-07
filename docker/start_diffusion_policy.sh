#!/usr/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

docker compose -f ${SCRIPT_DIR}/compose/robodiff_ros1.yml up -d