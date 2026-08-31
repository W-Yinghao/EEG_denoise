#!/bin/bash
# D4PM full-stage babysitter.
#
# The joint arm exceeds one 23:59:59 allocation; the adapter checkpoints on SIGUSR1 and
# exits 75, and the frozen route is "repeated full submissions of the same task index".
# This loop resubmits a task index only when (a) its result_summary.json is absent and
# (b) no job for that index is queued or running. It never cancels anything.
#
# Launched as a Slurm CPU job so it survives the interactive session.
set -uo pipefail
cd /home/infres/yinwang/denoiseNet

CONFIG=configs/baselines/d4pm_eog_scoped.yaml
RESULT_ROOT=results/cgdr/d4pm_benchmark/full/eog_scoped_seeded_native/eog
declare -A ARM=( [0]=joint_dual_diffusion [1]=matched_deterministic )
POLL=600
MAX_RESUBMITS=8
declare -A COUNT=( [0]=0 [1]=0 )

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

while true; do
    done_all=1
    for idx in 0 1; do
        summary="$RESULT_ROOT/${ARM[$idx]}/result_summary.json"
        if [[ -f "$summary" ]]; then
            continue
        fi
        done_all=0
        # is a job for this task index already queued or running?
        running_idx=$(squeue -u "$USER" -h -o "%K %j" 2>/dev/null \
                      | awk -v i="$idx" '$2=="dn-cgdr" && $1==i {print}' | wc -l)
        if [[ "$running_idx" -gt 0 ]]; then
            log "task $idx (${ARM[$idx]}) still active — waiting"
            continue
        fi
        if [[ "${COUNT[$idx]}" -ge "$MAX_RESUBMITS" ]]; then
            log "task $idx hit MAX_RESUBMITS=${MAX_RESUBMITS} — leaving it for a human"
            continue
        fi
        COUNT[$idx]=$(( COUNT[$idx] + 1 ))
        jid=$(scripts/slurm/submit.sh A100 cgdr --array "$idx" \
              d4pm-benchmark "$CONFIG" full 2>&1 | tail -1)
        log "resubmitted task $idx (${ARM[$idx]}) attempt ${COUNT[$idx]} -> $jid"
        sleep 30
    done
    if [[ "$done_all" -eq 1 ]]; then
        log "both arms have result_summary.json — submitting aggregate-full"
        scripts/slurm/submit.sh cpu cgdr d4pm-benchmark "$CONFIG" aggregate-full
        log "babysitter exiting"
        exit 0
    fi
    sleep "$POLL"
done
