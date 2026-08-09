#!/bin/bash
#SBATCH --job-name=taas_frz
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
ROOT=/home/infres/yinwang/denoiseNet_taas_subject_diffusion
PY=/home/infres/yinwang/anaconda3/envs/eeg2025/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
stage="${1:?stage}"
case "$stage" in
  assets)
    "$PY" scripts/taas_subject_diffusion_freeze.py
    ;;
  tests)
    "$PY" -m pytest -q tests/unit/test_taas_subject_diffusion_freeze.py tests/unit/test_bci2b_subject_diffusion_replication.py
    ;;
  compile)
    deps="$ROOT/.texmf/tex/latex/acmart"
    mkdir -p "$deps"
    if [[ ! -s "$deps/acmart.cls" || ! -s "$deps/ACM-Reference-Format.bst" ]]; then
      archive="${SLURM_TMPDIR:-/tmp}/acmart.zip"
      unpack="${SLURM_TMPDIR:-/tmp}/acmart_unpack"
      curl --fail --location --retry 3 --connect-timeout 20 --max-time 180 \
        "https://mirrors.ctan.org/macros/latex/contrib/acmart.zip" --output "$archive"
      mkdir -p "$unpack"
      unzip -q -o "$archive" -d "$unpack"
      ins="$(find "$unpack" -name acmart.ins -type f -print -quit)"
      (cd "$(dirname "$ins")" && latex -interaction=nonstopmode -halt-on-error "$(basename "$ins")")
      cp "$(find "$unpack" -name acmart.cls -type f -print -quit)" "$deps/acmart.cls"
      cp "$(find "$unpack" -name ACM-Reference-Format.bst -type f -print -quit)" "$deps/ACM-Reference-Format.bst"
    fi
    export TEXINPUTS="$deps:"
    export BSTINPUTS="$deps:"
    export TEXMFHOME="$ROOT/.texmf"
    timeout 300 tlmgr --ignore-warning --usermode --force \
      --repository https://ftp.tu-chemnitz.de/pub/tug/historic/systems/texlive/2024/tlnet-final \
      install xstring totpages environ trimspaces framed comment fontaxes mweights upquote \
      libertine newtx inconsolata multirow algorithmicx
    for package in hyperxmp ncctools; do
      case "$package" in
        hyperxmp) wanted=hyperxmp.sty ;;
        ncctools) wanted=manyfoot.sty ;;
      esac
      if ! kpsewhich "$wanted" >/dev/null 2>&1; then
        archive="${SLURM_TMPDIR:-/tmp}/${package}.zip"
        unpack="${SLURM_TMPDIR:-/tmp}/${package}_unpack"
        curl --fail --location --retry 3 --connect-timeout 20 --max-time 180 \
          "https://mirrors.ctan.org/macros/latex/contrib/${package}.zip" --output "$archive"
        mkdir -p "$unpack"
        unzip -q -o "$archive" -d "$unpack"
        while IFS= read -r ins; do
          (cd "$(dirname "$ins")" && latex -interaction=nonstopmode -halt-on-error "$(basename "$ins")")
        done < <(find "$unpack" -name '*.ins' -type f)
        find "$unpack" -name '*.sty' -type f -exec cp '{}' "$deps/" \;
      fi
    done
    cd taas_submission
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    bibtex main
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    ;;
  clean-check)
    snapshot="${SLURM_TMPDIR:-/tmp}/taas_clean_snapshot"
    index="${SLURM_TMPDIR:-/tmp}/taas_clean_index"
    mkdir -p "$snapshot"
    GIT_INDEX_FILE="$index" git read-tree HEAD
    GIT_INDEX_FILE="$index" git add -A
    tree="$(GIT_INDEX_FILE="$index" git write-tree)"
    git archive "$tree" | tar -x -C "$snapshot"
    PYTHONPATH="$snapshot/src:$snapshot" "$PY" -m pytest -q \
      "$snapshot/tests/unit/test_taas_subject_diffusion_freeze.py" \
      "$snapshot/tests/unit/test_bci2b_subject_diffusion_replication.py"
    PYTHONPATH="$snapshot/src:$snapshot" "$PY" -c \
      'import eeg_cgdr.models.eog_residual_diffusion; import eeg_cgdr.experiments.bci2b_subject_diffusion_replication; print("clean snapshot imports passed")'
    export TEXMFHOME="$ROOT/.texmf"
    export TEXINPUTS="$ROOT/.texmf/tex/latex/acmart:$ROOT/.texmf/tex/latex//:"
    export BSTINPUTS="$ROOT/.texmf/tex/latex/acmart:"
    cd "$snapshot/taas_submission"
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    bibtex main
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    test -s main.pdf
    ;;
  *)
    echo "unknown stage $stage" >&2
    exit 2
    ;;
esac
