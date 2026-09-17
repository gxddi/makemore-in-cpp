#!/usr/bin/env bash

# Setup the dependencies: libtorch, openapi (if applicable)

# Stop on errors, unset variables, and failed commands within pipelines.
set -euo pipefail

# Resolve paths relative to this script so it can be run from any directory.
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
deps_dir="${project_dir}/deps"
packages_dir="${project_dir}/packages"

# Detect an Intel display controller by default. USE_XPU=0 or USE_XPU=1 can
# override detection for containers, remote machines, or unsupported hardware.
case "${USE_XPU:-auto}" in
    1|on|true) use_xpu=1 ;;
    0|off|false) use_xpu=0 ;;
    auto)
        use_xpu=0
        if command -v lspci >/dev/null 2>&1 \
            && lspci | grep -Eiq 'VGA|3D|Display' \
            && lspci | grep -Ei 'VGA|3D|Display' | grep -Eiq 'Intel'; then
            use_xpu=1
        fi
        ;;
    *)
        echo "USE_XPU must be auto, 1, or 0" >&2
        exit 2
        ;;
esac

mkdir -p "${deps_dir}"

# TORCH_VERSION optionally pins the downloaded PyTorch/LibTorch version.
torch_requirement=torch
if [[ -n "${TORCH_VERSION:-}" ]]; then
    torch_requirement="torch==${TORCH_VERSION}"
fi

# PyTorch's XPU wheel pulls in the required oneAPI runtime wheels; the CPU wheel
# avoids those dependencies on machines without an Intel GPU.
if [[ "${use_xpu}" == 1 ]]; then
    torch_index=https://download.pytorch.org/whl/xpu
    echo "Intel GPU detected: installing LibTorch and oneAPI runtime"
else
    torch_index=https://download.pytorch.org/whl/cpu
    echo "No Intel GPU detected: installing CPU LibTorch"
fi

# Reinstall when switching backends. Otherwise, reuse the existing download.
backend=$([[ "${use_xpu}" == 1 ]] && echo xpu || echo cpu)
installed_backend=$(cat "${deps_dir}/backend" 2>/dev/null || true)
if [[ ! -f "${deps_dir}/torch/share/cmake/Torch/TorchConfig.cmake" \
    || "${installed_backend}" != "${backend}" ]]; then
    rm -rf "${packages_dir}"
    python3 -m pip install \
        --target "${packages_dir}" \
        --index-url "${torch_index}" \
        "${torch_requirement}"
    printf '%s\n' "${backend}" > "${deps_dir}/backend"
fi

# Extract libtorch into deps/
mv "${packages_dir}"/torch "${deps_dir}/torch"
if [[ "${use_xpu}" == 1 ]]; then
    # Intel runtime wheels place their lib/ and include/ trees here.
    ln -sfn packages "${deps_dir}/oneapi"
else
    rm -f "${deps_dir}/oneapi"
fi
