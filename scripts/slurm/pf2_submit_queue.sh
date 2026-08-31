#!/bin/bash
# Retry-until-accepted submitter for the E12/E3b/E4 + DS-DDPM chains.
# QOSMaxSubmitJobPerUserLimit means try again later, not failure.
LOG=/home/infres/yinwang/slurm_logs/paper_final/pf2_submit_queue.log
cd /home/infres/yinwang/denoiseNet
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
try() { # try NAME SBATCH [dependency]
  local name=$1 script=$2 dep=${3:-}
  local args=(--parsable)
  [ -n "$dep" ] && args+=("--dependency=$dep")
  local out
  out=$(sbatch "${args[@]}" "$script" 2>&1) && { say "$name accepted jid=$out"; echo "$out"; return 0; }
  say "$name rejected: $out"; echo ""; return 1
}
PROBE=""; TRAIN=""; ICA=""; E12=""; E3B=""; E4=""
for i in $(seq 1 480); do
  [ -z "$PROBE" ] && PROBE=$(try probe scripts/slurm/pf_dsddpm_probe.sbatch)
  [ -z "$E12" ]   && E12=$(try e12 scripts/slurm/pf_e12.sbatch)
  [ -z "$E3B" ]   && E3B=$(try e3b scripts/slurm/pf_e34_train.sbatch)
  [ -z "$ICA" ]   && ICA=$(try ica scripts/slurm/pf_dsddpm_ica.sbatch)
  [ -n "$PROBE" ] && [ -z "$TRAIN" ] && TRAIN=$(try dsddpm-train scripts/slurm/pf_dsddpm_train.sbatch "afterok:$PROBE")
  [ -n "$E12" ]   && [ -z "$E4" ]    && E4=$(try e4gen scripts/slurm/pf_e34_gen.sbatch "afterok:${E12}_0")
  if [ -n "$PROBE" ] && [ -n "$TRAIN" ] && [ -n "$ICA" ] && [ -n "$E12" ] && [ -n "$E3B" ] && [ -n "$E4" ]; then
    say "ALL SUBMITTED probe=$PROBE train=$TRAIN ica=$ICA e12=$E12 e3b=$E3B e4=$E4"; exit 0
  fi
  sleep 180
done
say "GAVE UP after 24h"; exit 1
