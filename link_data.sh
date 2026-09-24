#!/usr/bin/env bash
# Symlink every file in data/ to the same relative path under code/.
# The scripts assume code and data share one tree (e.g. common/items_sp_qwen35.jsonl),
# so after linking they run unchanged.
#   bash link_data.sh          # create links
#   bash link_data.sh --clean  # remove links
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(pwd)
LIST=code/.gitignore   # list of created links (also keeps them out of git)

if [[ "${1:-}" == "--clean" ]]; then
  if [[ -f $LIST ]]; then
    grep -v '^/\.gitignore$' "$LIST" | sed 's#^/##' | while read -r p; do if [[ -L "code/$p" ]]; then rm "code/$p"; fi; done
    rm -f "$LIST"
  fi
  echo "links removed"; exit 0
fi

echo "/.gitignore" > "$LIST"
n=0
while IFS= read -r -d '' f; do
  rel=${f#data/}
  case "$rel" in external/*) continue ;; esac
  dst="code/$rel"
  mkdir -p "$(dirname "$dst")"
  if [[ -e "$dst" && ! -L "$dst" ]]; then echo "skip (real file exists): $dst" >&2; continue; fi
  ln -sfn "$(realpath -s --relative-to="$(dirname "$dst")" "$f")" "$dst"
  echo "/$rel" >> "$LIST"; n=$((n+1))
done < <(find data \( -type f -o -type l \) -print0)

# Image paths inside the items ("cf2_images/...", "../data/external/...") are relative to code/.
# Some runners work from their own folder (code/occlusion, ...), so the same paths are linked there too.
ln -sfn ../data code/data; echo "/data" >> "$LIST"
for sub in common handoff2 mpreq occlusion optype; do
  for d in cf2_images; do
    [[ -e code/$d ]] || continue
    ln -sfn ../$d code/$sub/$d; echo "/$sub/$d" >> "$LIST"
  done
done

echo "linked $n files into code/  (run scripts from code/ or code/<subdir>/)"
