import sys, os, torch

ckpt_dir = sys.argv[1]
files = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pt"))
if not files:
    print(f"No .pt files in {ckpt_dir}"); sys.exit(1)
path = os.path.join(ckpt_dir, files[0])
print(f"Loading: {path}")
ckpt = torch.load(path, map_location="cpu", weights_only=False)

extra = ckpt.get("extra", {})
poe_stats = extra.get("poe_stats", {})
if not poe_stats:
    print("No poe_stats in checkpoint"); sys.exit(1)

print(f"\n{'Layer':>7}  {'ret_rms min':>12}  {'ret_rms max':>12}  {'ret_rms mean':>12}  {'C_hat frob':>12}  {'steps':>6}")
print("-" * 75)
for li in sorted(poe_stats.keys()):
    st = poe_stats[li]
    rms = st.get("ret_rms")
    C = st.get("C_hat")
    steps = st.get("_stats_steps", "?")
    if rms is not None:
        print(f"  L{li:02d}    {rms.min().item():.6e}  {rms.max().item():.6e}  {rms.mean().item():.6e}  {C.norm().item():.6e}  {steps:>6}")

for li in [0, max(poe_stats.keys())]:
    rms = poe_stats[li].get("ret_rms")
    if rms is not None:
        print(f"\nLayer {li} ret_rms per expert:")
        for e in range(rms.shape[0]):
            print(f"  expert {e:2d}: {rms[e].item():.8e}")
