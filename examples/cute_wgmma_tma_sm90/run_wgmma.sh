
#!/bin/bash
# run.sh

M=128
N=128
K=128

./cute_wgmma_tma_sm90 --M=${M} --N=${N} --K=${K} |& tee output.log
