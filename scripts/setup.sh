#!/usr/bin/env bash

# Setup the dependencies: libtorch, oneAPI (if applicable), nlohmann's json and sqlite3

# Stop on errors, unset variables, and failed commands within pipelines
set -euo pipefail

# Resolve paths relative to script
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
deps_dir=$project_dir/deps
cache_dir=$deps_dir/cache

mkdir -p "$deps_dir"
mkdir -p "$cache_dir"

# Detect Intel display controller, USE_XPU=0 or USE_XPU=1 can hardcode detection
case "${USE_XPU:-auto}" in
1|on|true) use_xpu=1 ;;
0|off|false) use_xpu=0 ;;
auto)
    lspci -D | grep -Ei 'VGA|3D|Display' |
        grep -i Intel |
        grep -qv '^0000:00:02\.0' &&
        use_xpu=1 || use_xpu=0
    ;;
*) echo "USE_XPU must be auto, 1, or 0" >&2; exit 2 ;;
esac

# TORCH_VERSION optionally pins the downloaded PyTorch/LibTorch version.
torch_requirement=torch
if [[ -n "${TORCH_VERSION:-}" ]]; then
    torch_requirement="torch==${TORCH_VERSION}"
fi

# PyTorch wheel index based on target backend
if [[ "${use_xpu}" == 1 ]]; then
    torch_index=https://download.pytorch.org/whl/xpu
    echo "Intel GPU detected: installing LibTorch and oneAPI runtime"
else
    torch_index=https://download.pytorch.org/whl/cpu
    echo "No Intel GPU detected: installing CPU LibTorch"
fi

# Install PyTorch into a temporary venv to preserve wheel structure
rm -rf "$cache_dir"/*
python3 -m venv "$cache_dir/venv"
"$cache_dir/venv/bin/pip" install --no-cache-dir "$torch_requirement" \
    --index-url "$torch_index"

# Locate site-packages inside venv
site_packages="$("$cache_dir/venv/bin/python" -c 'import site; print(site.getsitepackages()[0])')"

# Extract LibTorch into deps/
mkdir -p "$deps_dir/torch"
rm -rf "$deps_dir/torch"/*
mv "$site_packages/torch/include" "$deps_dir/torch/include"
mv "$site_packages/torch/lib" "$deps_dir/torch/lib"

# Extract oneAPI into deps/ (if applicable)
if [[ "${use_xpu}" == 1 ]]; then
    rm -rf "$deps_dir/oneapi"
    mkdir -p "$deps_dir/oneapi/lib" "$deps_dir/oneapi/include"

    # Copy oneAPI/SYCL headers (CL/, sycl/, oneapi/, umf/, etc.), skipping python headers
    find "$cache_dir/venv/include" -mindepth 1 -maxdepth 1 ! -name "python*" -exec cp -r {} "$deps_dir/oneapi/include/" \;

    # Filter only required runtime libraries and their dlopen dependencies
    wanted_libs=(
        # Explicitly linked by CMake
        "libsycl.so*"
        "libOpenCL.so*"
        "libpti_view.so*"
        "libmkl_sycl_blas.so*"
        "libmkl_sycl_dft.so*"
        "libmkl_sycl_lapack.so*"
        "libmkl_intel_lp64.so*"
        "libmkl_core.so*"
        "libmkl_gnu_thread.so*"

        # Hardware abstraction and runtime loaders
        "libur_loader.so*"
        "libur_adapter_level_zero*.so*"
        "libur_adapter_opencl.so*"
        "libxptifw.so*"
        "libiomp5.so*"
        "libtbb.so*"

        # MKL dispatch kernels
        "libmkl_def.so*"
        "libmkl_avx*.so*"
    )

    for pattern in "${wanted_libs[@]}"; do
        find "$cache_dir/venv/lib" -maxdepth 1 -name "$pattern" -exec mv {} "$deps_dir/oneapi/lib/" \;
    done

    # In your setup script, right after moving the oneAPI libraries:
    (
      cd "$deps_dir/oneapi/lib"
      for f in *.so.*; do
        base="${f%%.so.*}.so"
        [[ ! -e "$base" ]] && ln -s "$f" "$base"
      done
    )
else
    rm -rf "$deps_dir/oneapi"
fi

# Delete cache
rm -rf "$cache_dir"
