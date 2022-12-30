#!/usr/bin/env bash

# Script to install POCL (https://github.com/pocl/pocl).

set -ex

LLVM_VER=${LLVM_VER:-14}
POCL_VER=${POCL_VER:-3.1}
POCL_BUILD_DIR=${POCL_BUILD_DIR:-/tmp}

echo "Building POCL $POCL_VER with LLVM $LLVM_VER in $POCL_BUILD_DIR..."
echo "CMake flags: $CMAKE_FLAGS"

sudo apt-get update
sudo apt-get install -y --no-install-suggests --no-install-recommends \
  build-essential cmake pkg-config libhwloc-dev zlib1g-dev \
  ocl-icd-libopencl1 ocl-icd-dev ocl-icd-opencl-dev clinfo \
  clang-"$LLVM_VER" libclang-"$LLVM_VER"-dev llvm-"$LLVM_VER" \
  llvm-"$LLVM_VER"-dev libclang-cpp"$LLVM_VER" libclang-cpp"$LLVM_VER"-dev

if [[ ! -x "$POCL_BUILD_DIR/pocl-$POCL_VER/build/bin/poclcc" ]]; then
  cd "$POCL_BUILD_DIR" || exit 1

  # Get POCL sources
  curl -sSL "https://github.com/pocl/pocl/archive/v$POCL_VER.tar.gz" > pocl-"$POCL_VER".tar.gz
  tar xf "pocl-$POCL_VER.tar.gz"

  # Build POCL from the sources
  cd pocl-"$POCL_VER" || exit 1
  mkdir -p build && cd build || exit 1
  cmake -DWITH_LLVM_CONFIG=/usr/bin/llvm-config-"$LLVM_VER" \
    -DCMAKE_INSTALL_PREFIX=/usr $CMAKE_FLAGS ..
  make
fi

cd "$POCL_BUILD_DIR/pocl-$POCL_VER/build" && sudo make install

# Verify installation
clinfo
# If successful, `clinfo` should display information about the POCL platform.
