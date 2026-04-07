#!/bin/bash
TRACE_FILE="../../tracefile/gemm_general_2048.champsim.xz"
RESULTS_DIR="../../results_comparison"

echo "Running Verification for L2 Coalescing..."
./bin/my_config_mshr16_lfb10_lat10 --warmup_instructions 200000000 --simulation_instructions 500000000 $TRACE_FILE > $RESULTS_DIR/sector_l2.txt

echo "Done."
echo "Stats Comparison:"
echo "Baseline L2_RQ_FULL: $(grep "L2_RQ_FULL" $RESULTS_DIR/baseline.txt)"
echo "Sector V1 L2_RQ_FULL: $(grep "L2_RQ_FULL" $RESULTS_DIR/sector.txt)"
echo "Sector V2 L2_RQ_FULL: $(grep "L2_RQ_FULL" $RESULTS_DIR/sector_l2.txt)"
