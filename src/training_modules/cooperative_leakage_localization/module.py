from collections import OrderedDict
import numpy as np
import torch
from torch import nn, optim
import lightning as L
from scipy.stats import kendalltau, pearsonr

from common import *
from .utils import *
from ..utils import *
import utils.lr_schedulers
from utils.metrics import get_rank
from utils.gmm_performance_correlation import GMMPerformanceCorrelation
from utils.dnn_performance_auc import compute_dnn_performance_auc

class TemperaturePredictor(nn.Module):
    def __init__(self, input_len: int):
        super().__init__()
        self.input_len = input_len
        self.predictor = nn.Sequential(OrderedDict([
            ('dense_1', nn.Linear(self.input_len, 512)),
            ('act_1', nn.ReLU()),
            ('dense_2', nn.Linear(512, 1))
        ]))
    
    def forward(self, x):
        return 1 + self.predictor(x).exp()

class Module(L.LightningModule):
    def __init__(self,
        classifiers_name: str,
        classifiers_kwargs: dict = {},
        theta_lr_scheduler_name: str = None,
        theta_lr_scheduler_kwargs: dict = {},
        etat_lr_scheduler_name: str = None,
        etat_lr_scheduler_kwargs: dict = {},
        theta_lr: float = 1e-3,
        theta_beta_1: float = 0.9,
        etat_lr: float = 1e-3,
        etat_beta_1: float = 0.9,
        etat_beta_2: float = 0.999,
        etat_eps: float = 1e-8,
        etat_steps_per_theta_step: int = 1,
        theta_weight_decay: float = 0.0,
        etat_weight_decay: float = 0.0,
        ent_penalty: float = 0.0,
        starting_prob: float = 0.5,
        adversarial_mode: bool = True, ###########
        timesteps_per_trace: Optional[int] = None,
        class_count: int = 256,
        gradient_estimator: Literal['REINFORCE', 'REBAR'] = 'REBAR',
        rebar_relaxation: Literal['CONCRETE', 'MuProp'] = 'MuProp',
        noise_scale: Optional[float] = None,
        eps: float = 1e-6, # Constant that is added/subtracted from various things for numerical stability
        train_theta: bool = True,
        train_etat: bool = True,
        calibrate_classifiers: bool = False, # Should we do an online temperature calibration for the classifiers? Prevents overconfidence and makes val loss more-correlated w/ performance
        compute_gmm_ktcc: bool = False,
        reference_leakage_assessment: Optional[np.ndarray] = None,
        supervised_dnn: Optional[nn.Module] = None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        if self.hparams.theta_lr_scheduler_name is None:
            self.hparams.theta_lr_scheduler_name = 'NoOpLRSched'
        if self.hparams.etat_lr_scheduler_name is None:
            self.hparams.etat_lr_scheduler_name = 'NoOpLRSched'
        assert self.hparams.timesteps_per_trace is not None
        
        self.cmi_estimator = CondMutInfEstimator(
            self.hparams.classifiers_name,
            input_shape=(1, self.hparams.timesteps_per_trace),
            output_classes=self.hparams.class_count,
            classifiers_kwargs=self.hparams.classifiers_kwargs
        )
        self.selection_mechanism = SelectionMechanism(
            self.hparams.timesteps_per_trace,
            beta=self.hparams.starting_prob,
            adversarial_mode=self.hparams.adversarial_mode
        )
        if self.hparams.calibrate_classifiers:
            self.to_temperature = TemperaturePredictor(self.hparams.timesteps_per_trace)
        if self.hparams.gradient_estimator == 'REBAR':
            self.rebar_etat = nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=True)
            self.rebar_taut = nn.Parameter(torch.tensor(np.log(0.5), dtype=torch.float32), requires_grad=True)
        if not isinstance(self.hparams.reference_leakage_assessment, dict):
            if isinstance(self.hparams.reference_leakage_assessment, np.ndarray):
                self.hparams.reference_leakage_assessment = {'ref_0': self.hparams.reference_leakage_assessment}
            elif isinstance(self.hparams.reference_leakage_assessment, list):
                self.hparams.reference_leakage_assessment = {f'ref_{idx}': x for idx, x in enumerate(self.hparams.reference_leakage_assessment)}
            elif isinstance(self.hparams.reference_leakage_assessment, type(None)):
                pass
            else:
                assert False
        self.etat_step_counter = 0
    
    def to_global_steps(self, steps): # Lightning considers it a 'step' any time any optimizer is stepped. This converts training steps to Lightning steps (e.g. to pass to Trainer).
        out = 0
        if self.hparams.train_theta:
            out += steps
            if self.hparams.calibrate_classifiers:
                out += steps
        if self.hparams.train_etat:
            out += steps
            if self.hparams.gradient_estimator == 'REBAR':
                out += steps
        return out
    
    def rand_like(self, x):
        return self.hparams.eps + (1 - 2*self.hparams.eps)*torch.rand_like(x)
        
    def get_rebar_eta_and_tau(self):
        assert self.hparams.gradient_estimator == 'REBAR'
        rebar_eta = self.hparams.eps + 2*nn.functional.sigmoid(self.rebar_etat.exp())
        rebar_tau = self.hparams.eps + self.rebar_taut.exp()
        return rebar_eta, rebar_tau
    
    def configure_optimizers(self):
        self.etat_optimizer = optim.Adam(
            self.selection_mechanism.parameters(), lr=self.hparams.etat_lr, weight_decay=self.hparams.etat_weight_decay,
            betas=(self.hparams.etat_beta_1, self.hparams.etat_beta_2), eps=self.hparams.etat_eps
        )
        theta_yes_weight_decay, theta_no_weight_decay = [], []
        for name, param in self.cmi_estimator.named_parameters():
            if ('weight' in name) and not('norm' in name):
                theta_yes_weight_decay.append(param)
            else:
                theta_no_weight_decay.append(param)
        theta_param_groups = [{'params': theta_yes_weight_decay, 'weight_decay': self.hparams.theta_weight_decay}, {'params': theta_no_weight_decay, 'weight_decay': 0.0}]
        self.theta_optimizer = optim.AdamW(theta_param_groups, lr=self.hparams.theta_lr, betas=(self.hparams.theta_beta_1, 0.999))
        theta_lr_scheduler_constructor, etat_lr_scheduler_constructor = map(
            lambda x: (
                x if isinstance(x, (optim.lr_scheduler.LRScheduler))
                else getattr(utils.lr_schedulers, x) if hasattr(utils.lr_schedulers, x)
                else getattr(optim.lr_scheduler, x)
            ), (self.hparams.theta_lr_scheduler_name, self.hparams.etat_lr_scheduler_name)
        )
        if self.trainer.max_epochs != -1:
            self.total_steps = self.trainer.max_epochs*len(self.trainer.datamodule.train_dataloader())
        elif self.trainer.max_steps != -1:
            self.total_steps = self.trainer.max_steps
        else:
            assert False
        self.theta_lr_scheduler = theta_lr_scheduler_constructor(self.theta_optimizer, total_steps=self.total_steps, **self.hparams.theta_lr_scheduler_kwargs)
        self.etat_lr_scheduler = etat_lr_scheduler_constructor(self.etat_optimizer, total_steps=self.total_steps, **self.hparams.etat_lr_scheduler_kwargs)
        rv = [
            {'optimizer': self.theta_optimizer, 'lr_scheduler': {'scheduler': self.theta_lr_scheduler, 'interval': 'step'}},
            {'optimizer': self.etat_optimizer, 'lr_scheduler': {'scheduler': self.etat_lr_scheduler, 'interval': 'step'}}
        ]
        if self.hparams.calibrate_classifiers:
            self.to_temperature_optimizer = optim.Adam(self.to_temperature.parameters(), lr=self.hparams.theta_lr, betas=(0.9, 0.99999))
            self.to_temperature_lr_scheduler = utils.lr_schedulers.NoOpLRSched(self.to_temperature_optimizer)
            rv.append({'optimizer': self.to_temperature_optimizer, 'lr_scheduler': {'scheduler': self.to_temperature_lr_scheduler, 'interval': 'step'}})
        if self.hparams.gradient_estimator == 'REBAR':
            self.rebar_params_optimizer = optim.Adam([self.rebar_etat, self.rebar_taut], lr=self.hparams.etat_lr, betas=(0.9, 0.99999))
            self.rebar_params_lr_scheduler = etat_lr_scheduler_constructor(self.rebar_params_optimizer, total_steps=self.total_steps, **self.hparams.etat_lr_scheduler_kwargs)
            rv.append({'optimizer': self.rebar_params_optimizer, 'lr_scheduler': {'scheduler': self.rebar_params_lr_scheduler, 'interval': 'step'}})
        return rv

    def unpack_batch(self, batch):
        trace, label = batch
        assert trace.size(0) == label.size(0)
        if self.hparams.noise_scale is not None:
            trace += self.hparams.noise_scale*torch.randn_like(trace)
        return trace, label

    def get_b_values(self, trace: torch.Tensor, temperature: torch.Tensor):
        assert self.hparams.gradient_estimator == 'REBAR'
        if self.hparams.train_etat:
            log_gamma = self.selection_mechanism.get_log_gamma().unsqueeze(0)
            log_1mgamma = self.selection_mechanism.get_log_1mgamma().unsqueeze(0)
        else: # In the classifier pretraining stage it's convenient to clamp these to a specific value to make pretraining independent of our budget
            log_gamma = np.log(0.5)*torch.ones((1, *trace.shape[1:]), dtype=trace.dtype, device=trace.device)
            log_1mgamma = np.log(0.5)*torch.ones((1, *trace.shape[1:]), dtype=trace.dtype, device=trace.device)
        log_alpha = log_gamma - log_1mgamma
        assert torch.all(torch.isfinite(log_gamma))
        assert torch.all(torch.isfinite(log_1mgamma))
        u = self.rand_like(trace)
        b = torch.where(log_alpha + u.log() - (1-u).log() >= 0, torch.ones_like(u), torch.zeros_like(u))
        uprime = 1 - log_gamma.exp()
        v = self.rand_like(u)
        v = torch.where(b == 1, uprime + v*(1-uprime), v*uprime).clip_(self.hparams.eps, 1-self.hparams.eps)
        if self.hparams.rebar_relaxation == 'CONCRETE':
            to_z = lambda log_alpha, u: (log_alpha + u.log() - (1-u).log())
            rb = nn.functional.sigmoid(to_z(log_alpha, u)/temperature)
            rb_tilde = nn.functional.sigmoid(to_z(log_alpha, v)/temperature)
            rb_tilde_detached = nn.functional.sigmoid(to_z(log_alpha.detach(), v.detach())/temperature)
        elif self.hparams.rebar_relaxation == 'MuProp': # Like CONCRETE but modified so mean approaches that of Bernoulli distribution as temperature increases
            to_z = lambda log_alpha, u: ((temperature**2 + temperature + 1)/(temperature + 1))*log_alpha + u.log() - (1-u).log()
            rb = nn.functional.sigmoid(to_z(log_alpha, u)/temperature)
            rb_tilde = nn.functional.sigmoid(to_z(log_alpha, v)/temperature)
            rb_tilde_detached = nn.functional.sigmoid(to_z(log_alpha.detach(), v.detach())/temperature) # We only want to 'detach' with respect to log_gamma, not the temperature.
        else:
            raise NotImplementedError
        if self.hparams.adversarial_mode:
            b = 1-b
            rb = 1-rb
            rb_tilde = 1-rb_tilde
            rb_tilde_detached = 1-rb_tilde_detached
        return b, rb, rb_tilde, rb_tilde_detached
    
    def step(self, batch, train_theta: bool = True, train_etat: bool = True):
        if train_theta or train_etat:
            optimizers = self.optimizers()
            lr_schedulers = self.lr_schedulers()
            for optimizer in optimizers:
                optimizer.zero_grad()
            theta_optimizer = optimizers[0]
            theta_lr_scheduler = lr_schedulers[0]
            etat_optimizer = optimizers[1]
            etat_lr_scheduler = lr_schedulers[1]
            if self.hparams.calibrate_classifiers:
                to_temperature_optimizer = optimizers[2]
                to_temperature_lr_scheduler = lr_schedulers[2]
            if self.hparams.gradient_estimator == 'REBAR':
                rebar_params_optimizer = optimizers[-1]
                rebar_params_lr_scheduler = lr_schedulers[-1]
        if not train_theta:
            self.cmi_estimator.requires_grad_(False)
        if not train_etat:
            self.selection_mechanism.requires_grad_(False)
        trace, label = self.unpack_batch(batch)
        batch_size = trace.size(0)
        rv = {}
        if self.hparams.gradient_estimator == 'REINFORCE':
            raise NotImplementedError # Didn't seem to work well. I'm not going to bother implementing it now that I'm refactoring things.
        elif self.hparams.gradient_estimator == 'REBAR':
            rebar_eta, rebar_tau = self.get_rebar_eta_and_tau()
            b, rb, rb_tilde, rb_tilde_detached = self.get_b_values(trace, rebar_tau)
            bb = torch.cat([b, rb, rb_tilde, rb_tilde_detached], dim=0)
            logits = self.cmi_estimator.get_logits(trace.repeat(4, 1, 1), bb)
            theta_loss = nn.functional.cross_entropy(logits, label.repeat(4))
            rv.update({'theta_loss': theta_loss.detach()})
            with torch.no_grad():
                rv.update({'theta_rank': get_rank(logits, label.repeat(4)).mean()})
            if self.hparams.calibrate_classifiers:
                with torch.no_grad():
                    temperature = self.to_temperature(bb.flatten(start_dim=1))
                logits = logits/temperature
                theta_loss_calibrated = nn.functional.cross_entropy(logits, label.repeat(4))
                rv.update({'theta_loss_calibrated': theta_loss_calibrated.detach()})
                rv.update({'temperature': temperature.detach().cpu().numpy().mean()})
            mutinf = self.cmi_estimator.get_mutinf_estimate_from_logits(logits, label.repeat(4))
            mutinf_b, mutinf_rb, mutinf_rb_tilde, mutinf_rb_tilde_detached = map(lambda idx: mutinf[idx*batch_size:(idx+1)*batch_size], range(4))
            mutinf_b = mutinf_b.detach()
            log_p_b = self.selection_mechanism.log_pmf(b)
            etat_loss = -((mutinf_b - rebar_eta*mutinf_rb_tilde_detached)*log_p_b + rebar_eta*mutinf_rb - rebar_eta*mutinf_rb_tilde).mean()
            if self.hparams.adversarial_mode:
                etat_loss = -1*etat_loss
            if self.hparams.ent_penalty > 0:
                etat_loss = etat_loss + self.hparams.ent_penalty*(1 + log_p_b.detach().mean())*log_p_b.mean()
            rv.update({'etat_loss': etat_loss.detach()})
            rv.update({'hard_eta_loss': -mutinf_b.detach().cpu().numpy().mean()})
        else:
            assert False
        if train_theta:
            theta_loss.backward(retain_graph=train_etat, inputs=list(self.cmi_estimator.classifiers.parameters()))
        if train_etat:
            rv.update({'rebar_eta': rebar_eta.item(), 'rebar_tau': rebar_tau.item()})
            (etat_grad,) = torch.autograd.grad(etat_loss, self.selection_mechanism.etat, create_graph=True)
            self.selection_mechanism.etat.grad = etat_grad
            rv.update({'rms_grad': get_rms_grad(self.selection_mechanism)})
            if not hasattr(self, 'etat_grad_ema'):
                self.etat_grad_ema = etat_grad.detach()
            else:
                self.etat_grad_ema = 0.999*self.etat_grad_ema + 0.001*etat_grad.detach()
            rebar_params_loss = ((etat_grad - self.etat_grad_ema)**2).mean()
            (self.rebar_etat.grad, self.rebar_taut.grad) = torch.autograd.grad(rebar_params_loss, [self.rebar_etat, self.rebar_taut])
        if train_theta:
            theta_optimizer.step()
            theta_lr_scheduler.step()
        if train_etat:
            etat_optimizer.step()
            etat_lr_scheduler.step()
            rebar_params_optimizer.step()
            rebar_params_lr_scheduler.step()
            self.selection_mechanism.update_accumulated_gamma()
        if train_theta and self.hparams.calibrate_classifiers:
            if not(hasattr(self, 'cal_trace') and hasattr(self, 'cal_labels')): # don't want to spend a ton of time every step constructing dataloader, moving these to GPU, etc.
                val_trace, val_labels = next(iter(self.trainer.datamodule.val_dataloader()))
                val_trace = val_trace.to(self.device)
                val_labels = val_labels.to(self.device)
                self.cal_trace = val_trace
                self.cal_labels = val_labels
            with torch.no_grad():
                val_logits = self.cmi_estimator.get_logits(self.cal_trace[:batch_size, :, :].repeat(4, 1, 1), bb)
            temperature = self.to_temperature(bb.flatten(start_dim=1))
            val_loss = nn.functional.cross_entropy(val_logits/temperature, self.cal_labels[:batch_size].repeat(4)) + 1e-4*temperature.mean()**2
            val_loss.backward(inputs=list(self.to_temperature.parameters()))
            to_temperature_optimizer.step()
            to_temperature_lr_scheduler.step()
        self.cmi_estimator.requires_grad_(True)
        self.selection_mechanism.requires_grad_(True)
        assert all(torch.all(torch.isfinite(param)) for param in self.selection_mechanism.parameters())
        return rv
    
    def training_step(self, batch):
        if not(self.hparams.train_theta):
            train_theta = False
        elif self.hparams.train_theta and not(self.hparams.train_etat):
            train_theta = True
        else:
            self.etat_step_counter += 1
            if self.etat_step_counter <= self.hparams.etat_steps_per_theta_step:
                train_theta = True
                self.etat_step_counter = 0
            else:
                train_theta = False
        rv = self.step(batch, train_theta=train_theta, train_etat=self.hparams.train_etat)
        for key, val in rv.items():
            self.log(f'train_{key}', val, on_step=False, on_epoch=True)
    
    def validation_step(self, batch):
        rv = self.step(batch, train_theta=False, train_etat=False)
        for key, val in rv.items():
            self.log(f'val_{key}', val, on_step=False, on_epoch=True)
    
    def on_train_epoch_end(self):
        if self.hparams.reference_leakage_assessment is not None:
            gamma = self.selection_mechanism.get_accumulated_gamma().reshape(-1) #get_gamma().detach().cpu().numpy().reshape(-1)
            for key, leakage_assessment in self.hparams.reference_leakage_assessment.items():
                ktcc = kendalltau(gamma, leakage_assessment.reshape(-1)).statistic
                correlation = pearsonr(gamma, leakage_assessment.reshape(-1)).statistic
                self.log(f'{key}_ktcc', ktcc)
                self.log(f'{key}_corr', correlation)
        if (
            (self.total_steps // (100*len(self.trainer.train_dataloader)) == 0)
            or (self.current_epoch % (self.total_steps//(100*len(self.trainer.train_dataloader))) == 0)
        ):
            log_gamma = self.selection_mechanism.get_log_gamma().detach().cpu().numpy().squeeze()
            log_gamma_save_dir = os.path.join(self.logger.log_dir, 'log_gamma_over_time')
            os.makedirs(log_gamma_save_dir, exist_ok=True)
            np.save(os.path.join(log_gamma_save_dir, f'log_gamma__step={self.global_step}.npy'), log_gamma)
            
            if self.hparams.supervised_dnn is not None:
                gamma = self.selection_mechanism.get_accumulated_gamma().reshape(-1)
                dataloader = self.trainer.datamodule.val_dataloader()
                auc_results = compute_dnn_performance_auc(dataloader, self.hparams.supervised_dnn, gamma, device=self.device)
                for key, val in auc_results.items():
                    self.log(key, val)
                self.log('early_stop_metric', auc_results['reverse_dnn_auc'] - auc_results['forward_dnn_auc'])