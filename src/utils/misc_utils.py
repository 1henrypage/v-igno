# Stolen from DGenNO
import torch
import numpy as np

from pathlib import Path
import subprocess


def get_default_device() -> torch.device:
    """Return CUDA, MPS (Apple Silicon), or CPU in that order."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def setup_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_project_root() -> Path:
    return Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
        )
        .decode()
        .strip()
    )

def np2tensor(x:np.array, dtype=torch.float32, device: str | torch.device = get_default_device()):
    '''From numpy.array to torch.tensor
    '''
    return torch.tensor(x, dtype=dtype, device=device)

def detach2np(x:torch.tensor):
    '''Detach -> cpu -> numpy
    '''
    return x.detach().cpu().numpy()

def mesh1d(n, sub:int=1, low=0., high=1.):
    '''
    '''
    assert low<high
    assert sub<=n
    #
    mesh = np.linspace(low, high, n).reshape(-1,1)

    return mesh[::sub,:]

def mesh2d(nx, ny, subx:int=1, suby:int=1, xlow=0., xhigh=1., ylow=0., yhigh=1.):
    '''
    '''
    assert xlow<xhigh and ylow<yhigh
    assert subx<=nx and suby<=ny
    #
    x_mesh = np.linspace(xlow, xhigh, nx)[::subx]
    y_mesh = np.linspace(ylow, yhigh, ny)[::suby]
    xy_mesh = np.meshgrid(x_mesh, y_mesh)
    mesh = np.vstack([xy_mesh[0].flatten(), xy_mesh[1].flatten()]).T

    return mesh

if __name__=='__main__':
    mesh = mesh2d(nx=5, ny=5, subx=1, suby=1)
    print(mesh)

