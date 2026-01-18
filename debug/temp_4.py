import torch

# Original DGNO checkpoint

dgno_ckpt = torch.load('/home/henry/school/v-igno/runs/stable_darcy_continuous_igno/foundation/weights/best_dgno.pt',weights_only=False, map_location='cpu')

# Check w and b
for i in range(5):
    w_key = f'w.{i}'
    if w_key in dgno_ckpt['models']['a']:
        print(f"a.w.{i}: {dgno_ckpt['models']['a'][w_key]}")
        
print(f"a.b: {dgno_ckpt['models']['a']['b']}")
