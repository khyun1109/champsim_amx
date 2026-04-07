#!/bin/bash
set -e

# Configuration
CONFIG_NAME="jyh_config_mshr16_lfb10_lat10"
TRACE_FILE="../../tracefile/gemm_general_2048.champsim.xz"
BASE_DIR="/home/yhjeon/champsim"
SRC_DIR="$BASE_DIR/src/champsim"
HEADER_FILE="$SRC_DIR/inc/champsim.h"
RESULTS_DIR="$BASE_DIR/results_comparison"

mkdir -p $RESULTS_DIR

echo "=========================================="
echo "      Starting Comparison Experiment      "
echo "=========================================="

cd $SRC_DIR

# --- 1. Build Baseline (OFF) ---
echo "[1/4] Building Baseline (Sector MSHR OFF)..."
# Ensure the macro is NOT defined
sed -i '/#define ENABLE_SECTOR_MSHR/d' $HEADER_FILE

# Clean and Build
make clean >> /dev/null 2>&1 || true
# Run config script (adjust path if needed)
./config.sh ../../configs/$CONFIG_NAME.json
make -j$(nproc)
mv bin/my_config_mshr16_lfb10_lat10 bin/champsim_baseline
echo "      -> Built bin/champsim_baseline"

# --- 2. Build Sector MSHR (ON) ---
echo "[2/4] Building Sector MSHR (Sector MSHR ON)..."
# Define the macro
echo "#define ENABLE_SECTOR_MSHR" >> $HEADER_FILE

# Clean and Build
make clean >> /dev/null 2>&1 || true
./config.sh ../../configs/$CONFIG_NAME.json
make -j$(nproc)
mv bin/my_config_mshr16_lfb10_lat10 bin/champsim_sector
echo "      -> Built bin/champsim_sector"

# Cleanup Header
sed -i '/#define ENABLE_SECTOR_MSHR/d' $HEADER_FILE

# --- 3. Run Baseline ---
echo "[3/4] Running Baseline Simulation..."
echo "      Output: $RESULTS_DIR/baseline.txt"
./bin/champsim_baseline --warmup_instructions 200000000 --simulation_instructions 500000000 $TRACE_FILE > $RESULTS_DIR/baseline.txt &
PID_BASE=$!

# --- 4. Run Sector Mode ---
echo "[4/4] Running Sector MSHR Simulation..."
echo "      Output: $RESULTS_DIR/sector.txt"
./bin/champsim_sector --warmup_instructions 200000000 --simulation_instructions 500000000 $TRACE_FILE > $RESULTS_DIR/sector.txt &
PID_SECTOR=$!

wait $PID_BASE
wait $PID_SECTOR

echo "=========================================="
echo "          Comparison Completed            "
echo "=========================================="
echo "Results saved to $RESULTS_DIR:"
echo "  - baseline.txt"
echo "  - sector.txt"
grep "cumulative IPC" $RESULTS_DIR/baseline.txt | tail -1
grep "cumulative IPC" $RESULTS_DIR/sector.txt | tail -1
