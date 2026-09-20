"""Pick the N smallest mdCATH domain files and download them.

Size is a proxy for system size; small domains keep the probe cheap and the
recurrence question does not depend on chain length being large.
"""
import os, sys, json, urllib.request
from huggingface_hub import hf_hub_download

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
HAVE = {"1a02F00", "1a0aA00"}
DEST = os.path.expanduser("~/mdcath")

url = "https://huggingface.co/api/datasets/compsciencelab/mdCATH/tree/main/data?limit=10000"
rows = json.load(urllib.request.urlopen(url))
h5 = [(r["size"], r["path"]) for r in rows
      if r["path"].endswith(".h5")
      and r["path"].split("_")[-1][:-3] not in HAVE]
h5.sort()
pick = h5[:N]
print(f"{len(h5)} candidate domains; taking {len(pick)} smallest "
      f"({sum(s for s, _ in pick)/1e9:.2f} GB)\n")
for s, p in pick:
    print(f"  {s/1e6:7.0f} MB  {os.path.basename(p)}")
print()
for s, p in pick:
    out = hf_hub_download(repo_id="compsciencelab/mdCATH", repo_type="dataset",
                          filename=p, local_dir=DEST)
    print("ok", os.path.basename(out))
