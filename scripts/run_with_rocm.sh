#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export LIBTORCH=/usr
export LIBTORCH_BYPASS_VERSION_CHECK=1
export LD_LIBRARY_PATH=/usr/lib:/opt/rocm/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_PRELOAD=/usr/lib/libtorch_hip.so:/usr/lib/libc10_hip.so${LD_PRELOAD:+:${LD_PRELOAD}}
export ROCM_PATH=/opt/rocm

cargo run "$@"
