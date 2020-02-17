#!/bin/bash
set -e

python -m pip install -r requirements.txt

if [ "${#}" != 0 ]; then
  python -m pip install "${@}"
fi
