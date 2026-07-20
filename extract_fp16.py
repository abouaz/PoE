import torch, sys, os

ckpt_path = sys.argv[1]   # full checkpoint path
out_path  = sys.argv[2]   # output .pt path

print(f"Loading {ckpt_path} ...")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

model_sd = ckpt["model"]
fp16_sd = {k: v.half() for k, v in model_sd.items()}

print(f"Extracted {len(fp16_sd)} keys, saving to {out_path} ...")
torch.save(fp16_sd, out_path)

size_mb = os.path.getsize(out_path) / 1e6
print(f"Done. {size_mb:.0f} MB")
