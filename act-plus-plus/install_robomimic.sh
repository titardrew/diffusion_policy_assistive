#!/usr/bin/bash

git clone https://github.com/ARISE-Initiative/robomimic
cd robomimic
git fetch origin r2d2
git checkout r2d2
pip install -e .
