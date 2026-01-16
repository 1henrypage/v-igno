import torch

# Check what's in the pretrained checkpoint
ckpt = torch.load('/Users/henry/school/v-igno/runs/stable_dgno_darcy_continuous/foundation/weights/best_dgno.pt', 
                   weights_only=False, map_location='cpu')

print("Models in checkpoint:", list(ckpt['models'].keys()))

if 'nf' in ckpt['models']:
    print("\nNF is in checkpoint - it will be loaded!")
    nf_state = ckpt['models']['nf']
    for k in nf_state:
        if 'log_scale_base' in k:
            print(f"  {k}: {nf_state[k].mean():.4f}")
