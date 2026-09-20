set -e
pip install -U "huggingface_hub[cli]" h5py

mkdir -p ~/mdcath && cd ~/mdcath

for D in 1a02F00 1a0aA00 153lA00 16pkA02 12asA00; do
  hf download compsciencelab/mdCATH \
      --repo-type dataset \
      --include "data/mdcath_dataset_${D}.h5" \
      --local-dir ~/mdcath
done

du -sh ~/mdcath/data

cd ~/SciML-MD
python validation/phase8_recurrence.py \
    --mdcath "$HOME/mdcath/data/*.h5" \
    --temps 320 450 \
    --out validation/phase8_recurrence_mdcath.json
