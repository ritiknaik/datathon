#!/bin/bash
# Run on the Grace LOGIN node after grace_job has finished (light work: copy + git push).
# Saves the trained weights + logs into models/grace_<date>/ and pushes them to spoc_db,
# so the team can compare them locally without logging in to Grace again.
#   bash train/grace_save.sh
# The weights are NOT put in bot/: the shipped bot stays v2 until compare.py accepts the new net.
set -e
D=$SCRATCH/pokerdata
cd $SCRATCH/datathon
ls $D/q_weights.npz $D/q_weights_no_nit.npz >/dev/null || { echo "weights missing: is grace_job finished?"; exit 1; }
[ -n "$(git config user.email)" ] || { echo 'first run: git config user.name "Your Name" && git config user.email "you@tamu.edu"'; exit 1; }

RUN=models/grace_$(date +%Y%m%d_%H%M)
mkdir -p $RUN
cp $D/q_weights.npz $D/q_weights_no_nit.npz $RUN/
cp "$(ls -t $D/logs/grace_job.*.out | head -1)" $RUN/train.log
grep -h saved $D/logs/gen.*.out > $RUN/data.log || true
echo "rows: $(awk '{s+=$2} END {print s}' $RUN/data.log)  files: $(wc -l < $RUN/data.log)" | tee $RUN/summary.txt

read -rsp "GitHub token (needs Contents: Read and write; not saved): " TOKEN; echo
URL=https://$TOKEN@github.com/ritiknaik/datathon.git
git add $RUN
git commit -q -m "Add Grace-trained Q-net weights ($RUN)"
git pull -q --rebase $URL spoc_db
git push -q $URL HEAD:spoc_db
unset TOKEN URL
echo "pushed $RUN to spoc_db: $(git log --oneline -1)"
