#!/bin/bash

VENV_PATH="${VENV_PATH:-$(realpath "$HOME")/venvs}"
VENV_NAME="${VENV_NAME:-cpvae}"

if [ ! -d "${VENV_PATH}/${VENV_NAME}" ]; then
  echo "Creating virtualenv"
  python3 -m venv "${VENV_PATH}/${VENV_NAME}"
fi

source "${VENV_PATH}/${VENV_NAME}/bin/activate"

#./scripts/install-deps.sh "dvc[s3]" --upgrade
./scripts/install-deps.sh
#source ./scripts/read-env-file.sh
