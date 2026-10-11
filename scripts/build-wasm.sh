#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODULE_DIR="${ROOT_DIR}/zixy-py"
EMSDK_DIR="${EMSDK_DIR:-${ROOT_DIR}/.emsdk}"
EMSDK_VERSION="${EMSDK_VERSION:-4.0.9}"
PYODIDE_ABI_VERSION="${PYODIDE_ABI_VERSION:-2025_0_28}"
RUST_TOOLCHAIN="${RUST_TOOLCHAIN:-nightly}"
TARGET="wasm32-unknown-emscripten"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/dist}"

for command_name in git rustup uv; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        printf 'error: required command not found: %s\n' "${command_name}" >&2
        exit 1
    fi
done

PYTHON_BIN="${PYTHON_BIN:-$(uv python find 3.12)}"
export PATH="$(dirname "$(command -v "${PYTHON_BIN}")"):${PATH}"

if [[ ! -d "${MODULE_DIR}" ]]; then
    printf 'error: Python module directory not found: %s\n' "${MODULE_DIR}" >&2
    exit 1
fi

rustup toolchain install "${RUST_TOOLCHAIN}"
rustup target add "${TARGET}" --toolchain "${RUST_TOOLCHAIN}"
rustup component add rust-src --toolchain "${RUST_TOOLCHAIN}"

if [[ ! -x "${EMSDK_DIR}/emsdk" ]]; then
    if [[ -e "${EMSDK_DIR}" ]]; then
        printf 'error: emsdk path exists but is not an emsdk checkout: %s\n' "${EMSDK_DIR}" >&2
        exit 1
    fi
    git clone --depth 1 https://github.com/emscripten-core/emsdk.git "${EMSDK_DIR}"
fi

"${EMSDK_DIR}/emsdk" install "${EMSDK_VERSION}"
"${EMSDK_DIR}/emsdk" activate "${EMSDK_VERSION}"

mkdir -p "${OUTPUT_DIR}"

cd "${MODULE_DIR}"
# shellcheck disable=SC1091
source "${EMSDK_DIR}/emsdk_env.sh"

env \
    CARGO_BUILD_TARGET="${TARGET}" \
    TARGET="${TARGET}" \
    CFLAGS="-fPIC" \
    CXXFLAGS="-fPIC" \
    RUSTUP_TOOLCHAIN="${RUST_TOOLCHAIN}" \
    MATURIN_PYEMSCRIPTEN_PLATFORM_VERSION="${PYODIDE_ABI_VERSION}" \
    uvx maturin build \
        -Z build-std \
        --release \
        -i "${PYTHON_BIN}" \
        --target "${TARGET}" \
        -o "${OUTPUT_DIR}" \
        -v

printf 'Wasm wheel(s) written to %s\n' "${OUTPUT_DIR}"