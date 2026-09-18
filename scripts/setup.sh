#!/usr/bin/env bash

# Setup the dependencies: libtorch, openapi (if applicable)

# Stop on errors, unset variables, and failed commands within pipelines
set -euo pipefail

# Resolve paths relative to script
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
deps_dir="${project_dir}/deps"
packages_dir="${project_dir}/packages"

# Detect Intel display controller, USE_XPU=0 or USE_XPU=1 can hardcode detection
case "${USE_XPU:-auto}" in
1|on|true) use_xpu=1 ;;
0|off|false) use_xpu=0 ;;
auto)
    use_xpu=0
    for d in /sys/class/drm/card*/device; do
        [ "$(cat "$d/vendor" 2>/dev/null)" = "0x8086" ] || continue
        [ "$(cat "$d/mem_info_vram_total" 2>/dev/null)" -gt 0 ] 2>/dev/null && {
            use_xpu=1
            break
        }
    done
    ;;
*) echo "USE_XPU must be auto, 1, or 0" >&2; exit 2 ;;
esac

mkdir -p "${deps_dir}"

# TORCH_VERSION optionally pins the downloaded PyTorch/LibTorch version.
torch_requirement=torch
if [[ -n "${TORCH_VERSION:-}" ]]; then
    torch_requirement="torch==${TORCH_VERSION}"
fi

# PyTorch's XPU wheel pulls in the required oneAPI runtime
if [[ "${use_xpu}" == 1 ]]; then
    torch_index=https://download.pytorch.org/whl/xpu
    echo "Intel GPU detected: installing LibTorch and oneAPI runtime"
else
    torch_index=https://download.pytorch.org/whl/cpu
    echo "No Intel GPU detected: installing CPU LibTorch"
fi

# Install with pip
rm -rf "${packages_dir}"
python3 -m pip install \
        --target "${packages_dir}" \
        --index-url "${torch_index}" \
        "${torch_requirement}"

# Extract libtorch into deps/
mkdir -p "${deps_dir}/torch"
rm -rf "${deps_dir}/torch/*"
mv "${packages_dir}/torch/include" "${deps_dir}/torch/include"
mv "${packages_dir}/torch/lib" "${deps_dir}/torch/lib"

# Extrach OneApi if it it's used
if [[ "${use_xpu}" == 1 ]]; then
    mkdir -p "${deps_dir}/oneapi"
    rm -rf "${deps_dir}/oneapi/*"
    mv "${packages_dir}"/* "${deps_dir}/oneapi/"
else
    rm -rf "${deps_dir}/oneapi"
fi
