#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
python3 main.py --config ConfigV3_Stage2_CommonTest > logs/run_${TIMESTAMP}.out > logs/run_${TIMESTAMP}.err