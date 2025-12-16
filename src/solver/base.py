
import torch
import torch.nn as nn
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Literal
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path


from src.solver.config import SchedulerConfig, OptimizerConfig, LossWeights
from src.utils.Losses import MyError, MyLoss
from src.utils.misc_utils import get_default_device, get_project_root
from src.utils.solver_utils import get_optimizer, get_scheduler, data_loader

ARTIFACT_DIR: Path = get_project_root() / "runs"


# Loss classes
#############################
class ProblemInstance(ABC):
    """Abstract base class for PDE losses"""

    def __init__(
            self,
            device: torch.device | str = get_default_device(),
            dtype: Optional[torch.dtype] = torch.float32,
    ) -> None:
        self.device = device
        self.dtype = dtype

        # INIT
        self.init_error()
        self.init_loss()

        # Forward declare
        self.get_loss = None
        self.get_error = None

    @abstractmethod
    def loss_beta(
            self,
            a: torch.Tensor
    ) -> Optional[torch.Tensor]:
        return 0

    @abstractmethod
    def loss_pde(
            self,
            a: torch.Tensor
    ) -> torch.Tensor:
        return 0

    @abstractmethod
    def loss_data(
            self,
            x: torch.Tensor,
            a: torch.Tensor,
            u: torch.Tensor
    ) -> torch.Tensor:
        return 0

    @abstractmethod
    def get_model_dict(self) -> Dict[str, nn.Module]:
        pass

    @abstractmethod
    def error(
            self,
            x: torch.Tensor,
            a: torch.Tensor,
            u: torch.Tensor
    ) -> torch.Tensor:
        return 0

    def init_error(
            self,
            err_type: str='lp_rel',
            d: int = 2,
            p: int = 2,
            size_average: bool = True,
            reduction: bool = True,
    ) -> None:
        self.get_error = MyError(
            d=d,
            p=p,
            size_average=size_average,
            reduction=reduction,
        )(err_type)

    def init_loss(
            self,
            loss_type: str = 'mse_org',
            size_average=True,
            reduction=True
    ):
        self.get_loss = MyLoss(
            size_average=size_average,
            reduction=reduction
        )(loss_type)

    def pre_train_check(self) -> None:
        """
        This is useful if we don't do any custom losses.
        The solver will initialise the default settings.

        Logic:
        - If both get_loss and get_error are None, call init_loss() and init_error()
        - If exactly one of them is None, raise an error
        - Otherwise, do nothing
        """
        loss_none = self.get_loss is None
        error_none = self.get_error is None

        if loss_none and error_none:
            # Initialize both
            self.init_loss()
            self.init_error()
        elif loss_none != error_none:  # XOR: exactly one is None
            raise ValueError("Both get_loss and get_error must be set, or both None to auto-initialize.")
        # else: both are already set, do nothing




class Solver:
    """
    Base class for all solvers.
    """

    def __init__(
            self,
            problem_instance: ProblemInstance,
    ):
        self.problem_instance = problem_instance
        self.device = problem_instance.device
        self.dtype = problem_instance.dtype

        # Forward declare
        self.model_dict = None
        self.optimizer= None
        self.scheduler = None
        self.writer: SummaryWriter = None
        self.weights_dir = None
        self.tb_dir = None

    def init_logging(
            self,
            loss_weights: LossWeights,
            optimizer_config: OptimizerConfig,
            scheduler_config: SchedulerConfig,
            custom_run_tag: str = None
    ) -> None:

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_name = f"{timestamp}_{custom_run_tag}" if custom_run_tag else timestamp
        run_dir = ARTIFACT_DIR / run_name
        self.weights_dir = run_dir / "weights"
        self.tb_dir = run_dir / "tb"
        self.weights_dir.mkdir(parents=True)
        self.tb_dir.mkdir(parents=True)

        # Save config
        (loss_weights.save(run_dir / "loss_weights.yaml"))
        (optimizer_config.save(run_dir / "optimizer_config.yaml"))
        (scheduler_config.save(run_dir / "scheduler_config.yaml"))

        self.writer = SummaryWriter(
            log_dir=str(self.tb_dir)
        )

    def log_epoch(
            self,
            loss_train: torch.Tensor,
            loss_data: torch.Tensor,
            loss_pde: torch.Tensor,
            loss_test: torch.Tensor,
            error_test: torch.Tensor,
            t_start: float,
            epoch: int
    ):
        self.writer.add_scalar("train/loss_train", loss_train.item(), epoch)
        self.writer.add_scalar("train/loss_data", loss_data.item(), epoch)
        self.writer.add_scalar("train/loss_pde", loss_pde.item(), epoch)

        # --- Log test loss ---
        self.writer.add_scalar("test/loss", loss_test.item(), epoch)
        self.writer.add_scalar("train/time", time.time() - t_start)

        # --- Log test error ---
        if error_test.numel() > 1:
            for i, err in enumerate(error_test):
                self.writer.add_scalar(f"test/error_{i}", err.item(), epoch)
        else:
            self.writer.add_scalar("test/error", error_test.item(), epoch)

    def save_models(self, filename: str):
        """
        Saves the model weights to self.weights_dir with the given filename.
        """
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.weights_dir / filename

        # Extract state_dicts
        state_dicts = {name: model.state_dict() for name, model in self.model_dict.items()}
        torch.save(state_dicts, save_path)

    def load_models(self, filename: str):
        load_path = self.weights_dir / filename
        if not load_path.exists():
            raise FileNotFoundError(f"No model file found at {load_path}")

        state_dicts = torch.load(load_path, map_location=self.device)
        for name, model in self.model_dict.items():
            if name in state_dicts:
                model.load_state_dict(state_dicts[name])
            else:
                raise RuntimeError(f"No model state dict found for {name}")

    def pre_train_init(
            self,
            scheduler_config: SchedulerConfig,
            optimizer_config: OptimizerConfig
    ) -> None:
        self.model_dict = self.problem_instance.get_model_dict()
        param_list = []

        for k in self.model_dict:
            self.model_dict[k] = self.model_dict[k].to(self.device)


        for model in self.model_dict.values():
            param_list += list(model.parameters())

        self.optimizer = get_optimizer(
            optimizer_config=optimizer_config,
            param_list=param_list,
        )

        self.scheduler = get_scheduler(
            scheduler_config=scheduler_config,
            optimizer=self.optimizer,
        )

    def activate_train(self) -> None:
        for model in self.model_dict.values():
            model.train()

    def activate_eval(self) -> None:
        for model in self.model_dict.values():
            model.eval()

    def scheduler_step(
            self,
            error_test: torch.Tensor,
            scheduler_config: SchedulerConfig
    ) -> None:
        scheduler_type = scheduler_config.type

        if scheduler_type is None:
            pass
        elif scheduler_type=='Plateau':
            self.scheduler.step(error_test.item())
        else:
            self.scheduler.step()


    def train_mbgd(
            self,
            a_train, u_train, x_train,
            a_test, u_test, x_test,
            optimizer_config: OptimizerConfig = OptimizerConfig(),
            scheduler_config: SchedulerConfig = SchedulerConfig(),
            loss_weights: LossWeights = LossWeights(),
            batch_size: int = 100,
            epochs: int = 1,
            epoch_show: int = 10,
            shuffle=True,
            custom_run_tag: str = '',
            **kwrds
    ):

        self.problem_instance.pre_train_check()

        # ---- Generate run name ----
        self.init_logging(
            loss_weights=loss_weights,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            custom_run_tag=custom_run_tag
        )

        # INIT
        self.pre_train_init(
            scheduler_config=scheduler_config,
            optimizer_config=optimizer_config
        )

        # Loaders
        train_loader = data_loader(
            a=a_train,
            u=u_train,
            x=x_train,
            batch_size=batch_size,
            shuffle=shuffle,
        )

        test_loader = data_loader(
            a=a_test,
            u=u_test,
            x=x_test,
            batch_size=batch_size
        )

        t_start = time.time()
        best_err_test = 1e10

        # Training
        for epoch in trange(epochs):

            self.activate_train()
            loss_train_sum, loss_data_sum, loss_pde_sum, loss_beta_sum = 0., 0., 0., 0.

            for a, u, x in train_loader:
                a,u,x = a.to(self.device), u.to(self.device), x.to(self.device)

                loss_pde = self.problem_instance.loss_pde(a)
                loss_data = self.problem_instance.loss_data(x,a,u)
                loss_beta = self.problem_instance.loss_beta(a)
                loss_train = loss_pde * loss_weights.pde+ loss_data * loss_weights.data+ loss_beta * loss_weights.beta

                self.optimizer.zero_grad()
                loss_train.backward()
                self.optimizer.step()

                loss_train_sum += loss_train
                loss_data_sum += loss_data
                loss_pde_sum += loss_pde
                loss_beta_sum += loss_beta

            self.activate_eval()
            loss_test_sum = 0
            error_test_sum = 0
            for a, u, x in test_loader:
                a, u, x = a.to(self.device), u.to(self.device), x.to(self.device)
                with torch.no_grad():
                    loss_test = self.problem_instance.loss_data(x,a,u)
                    error_test = self.problem_instance.error(x,a,u)
                loss_test_sum += loss_test
                error_test_sum += error_test

            self.log_epoch(
                loss_train=loss_train_sum/len(train_loader),
                loss_data=loss_data_sum/len(train_loader),
                loss_pde=loss_pde_sum/len(train_loader),
                loss_test=loss_test_sum/len(test_loader),
                error_test=error_test_sum/len(test_loader),
                t_start=t_start,
                epoch=epoch,
            )

            error_test = torch.mean(error_test_sum/len(test_loader))
            if error_test.item() < best_err_test:
                best_err_test = error_test.item()
                self.save_models(filename='best.pt')

            # Possible scheduler step
            self.scheduler_step(
                error_test=error_test,
                scheduler_config=scheduler_config
            )

            if (epoch + 1) % epoch_show == 0:
                print(
                    f'Epoch:{epoch + 1} Time:{time.time() - t_start:.4f}, loss:{loss_train_sum.item() / len(train_loader):.4f}, loss_pde:{loss_pde_sum.item() / len(train_loader):.4f}, loss_data:{loss_data_sum.item() / len(train_loader):.4f}')
                for para in self.optimizer.param_groups:
                    print(f"                l2_test:{error_test.item():.4f}, lr:{para['lr']}")

        self.save_models(filename='last.pt')

        print(f'The total training time is {time.time() - t_start:.4f}')

