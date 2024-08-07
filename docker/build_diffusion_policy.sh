#!/usr/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo ${SCRIPT_DIR}
docker build -t cuda_11_6_ros1_20_04_base -f ${SCRIPT_DIR}/Dockerfile_ros1_20_04 ${SCRIPT_DIR}/..
docker compose -f ${SCRIPT_DIR}/compose/robodiff_ros1.yml build