from typing import Any, Mapping, Optional, Sequence, Union

import copy
import itertools
import gymnasium
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.multi_agents.torch import MultiAgent, MultiAgentIrat
from skrl.resources.schedulers.torch import KLAdaptiveLR


# fmt: off
# [start-config-dict-torch]
IRAT_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "learning_epochs": 8,           # number of learning epochs during each update
    "mini_batches": 2,              # number of mini batches during each learning epoch

    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages

    "learning_rate": 1e-3,                  # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "shared_state_preprocessor": None,      # shared state preprocessor class (see skrl.resources.preprocessors)
    "shared_state_preprocessor_kwargs": {}, # shared state preprocessor's kwargs (e.g. {"size": env.shared_observation_space})
    "value_preprocessor": None,             # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},        # value preprocessor's kwargs (e.g. {"size": 1})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0.5,              # clipping coefficient for the norm of the gradients
    "ratio_clip": 0.2,                  # clipping coefficient for computing the clipped surrogate objective
    "idv_ratio_clip": 0.5,                  # clipping coefficient for computing the clipped surrogate objective
    "value_clip": 0.2,                  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    "clip_predicted_values": False,     # clip predicted values during value loss computation

    "entropy_loss_scale": 0.0,      # entropy loss scaling factor
    "value_loss_scale": 1.0,        # value loss scaling factor

    "kl_threshold": 0,              # KL divergence threshold for early stopping

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)

    "mixed_precision": False,       # enable automatic mixed precision for higher performance

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}
# [end-config-dict-torch]
# fmt: on


class IRAT(MultiAgentIrat):
    def __init__(
        self,
        possible_agents: Sequence[str],
        models: Mapping[str, Model],
        memories: Optional[Mapping[str, Memory]] = None,
        observation_spaces: Optional[Union[Mapping[str, int], Mapping[str, gymnasium.Space]]] = None,
        action_spaces: Optional[Union[Mapping[str, int], Mapping[str, gymnasium.Space]]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
        shared_observation_spaces: Optional[Union[Mapping[str, int], Mapping[str, gymnasium.Space]]] = None,
    ) -> None:
        """Heterogeneous-Agent Proximal Policy Optimization (IRAT)

        https://arxiv.org/abs/2103.01955

        :param possible_agents: Name of all possible agents the environment could generate
        :type possible_agents: list of str
        :param models: Models used by the agents.
                       External keys are environment agents' names. Internal keys are the models required by the algorithm
        :type models: nested dictionary of skrl.models.torch.Model
        :param memories: Memories to storage the transitions.
        :type memories: dictionary of skrl.memory.torch.Memory, optional
        :param observation_spaces: Observation/state spaces or shapes (default: ``None``)
        :type observation_spaces: dictionary of int, sequence of int or gymnasium.Space, optional
        :param action_spaces: Action spaces or shapes (default: ``None``)
        :type action_spaces: dictionary of int, sequence of int or gymnasium.Space, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict
        :param shared_observation_spaces: Shared observation/state space or shape (default: ``None``)
        :type shared_observation_spaces: dictionary of int, sequence of int or gymnasium.Space, optional
        """
        _cfg = copy.deepcopy(IRAT_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(
            possible_agents=possible_agents,
            models=models,
            memories=memories,
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            device=device,
            cfg=_cfg,
        )

        self.shared_observation_spaces = shared_observation_spaces

        # Individual models
        self.idv_policies = {uid: self.idv_models[uid].get("policy", None) for uid in self.possible_agents}
        self.idv_values = {uid: self.idv_models[uid].get("value", None) for uid in self.possible_agents}
        
        # team models
        self.team_policies = {uid: self.team_models[uid].get("policy", None) for uid in self.possible_agents}
        self.team_values = {uid: self.team_models[uid].get("value", None) for uid in self.possible_agents}

        for uid in self.possible_agents:
            # checkpoint models
            self.checkpoint_modules[uid]["idv_policy"] = self.idv_policies[uid]
            self.checkpoint_modules[uid]["idv_value"] = self.idv_values[uid]
            self.checkpoint_modules[uid]["team_policy"] = self.team_policies[uid]
            self.checkpoint_modules[uid]["team_value"] = self.team_values[uid]
            
            # broadcast models' parameters in distributed runs
            if config.torch.is_distributed:
                logger.info(f"Broadcasting models' parameters")
                if self.idv_policies[uid] is not None:
                    self.idv_policies[uid].broadcast_parameters()
                    if self.idv_values[uid] is not None and self.idv_policies[uid] is not self.idv_values[uid]:
                        self.idv_values[uid].broadcast_parameters()
                if self.team_policies[uid] is not None:
                    self.team_policies[uid].broadcast_parameters()
                    if self.team_values[uid] is not None and self.team_policies[uid] is not self.team_values[uid]:
                        self.team_values[uid].broadcast_parameters()

        # configuration
        self._learning_epochs = self._as_dict(self.cfg["learning_epochs"])
        self._mini_batches = self._as_dict(self.cfg["mini_batches"])
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0

        self._grad_norm_clip = self._as_dict(self.cfg["grad_norm_clip"])
        self._ratio_clip = self._as_dict(self.cfg["ratio_clip"])
        self._value_clip = self._as_dict(self.cfg["value_clip"])
        self._clip_predicted_values = self._as_dict(self.cfg["clip_predicted_values"])

        self._value_loss_scale = self._as_dict(self.cfg["value_loss_scale"])
        self._entropy_loss_scale = self._as_dict(self.cfg["entropy_loss_scale"])

        self._kl_threshold = self._as_dict(self.cfg["kl_threshold"])

        self._learning_rate = self._as_dict(self.cfg["learning_rate"])
        self._learning_rate_scheduler = self._as_dict(self.cfg["learning_rate_scheduler"])
        self._learning_rate_scheduler_kwargs = self._as_dict(self.cfg["learning_rate_scheduler_kwargs"])

        self._state_preprocessor = self._as_dict(self.cfg["state_preprocessor"])
        self._state_preprocessor_kwargs = self._as_dict(self.cfg["state_preprocessor_kwargs"])
        self._shared_state_preprocessor = self._as_dict(self.cfg["shared_state_preprocessor"])
        self._shared_state_preprocessor_kwargs = self._as_dict(self.cfg["shared_state_preprocessor_kwargs"])
        self._value_preprocessor = self._as_dict(self.cfg["value_preprocessor"])
        self._value_preprocessor_kwargs = self._as_dict(self.cfg["value_preprocessor_kwargs"])

        self._discount_factor = self._as_dict(self.cfg["discount_factor"])
        self._lambda = self._as_dict(self.cfg["lambda"])

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._rewards_shaper = self.cfg["rewards_shaper"]
        self._time_limit_bootstrap = self._as_dict(self.cfg["time_limit_bootstrap"])

        self._mixed_precision = self.cfg["mixed_precision"]
        
        # IRAT act
        self._act_policy = self.cfg["act_policy"]  # idv or team
        
        # IRAT LOSS
        self._idv_use_cross_entropy = self.cfg.get("idv_use_cross_entropy", False)
        self._team_use_cross_entropy = self.cfg.get("team_use_cross_entropy", False)
        self._idv_clip_use_present = self.cfg.get("idv_clip_use_present", False)
        self._team_clip_use_present = self.cfg.get("team_clip_use_present", False)
        self._idv_use_two_clip = self.cfg.get("idv_use_two_clip", True)
        self._idv_use_kl_loss = self.cfg.get("idv_use_kl_loss", True)
        self._team_use_kl_loss = self.cfg.get("team_use_kl_loss", True)
        self._team_use_clip = self.cfg.get("team_use_clip", True)
        
        self._idv_ratio_clip = self._as_dict(self.cfg["idv_ratio_clip"])
        self._idv_kl_coef = self._as_dict(self.cfg.get("idv_kl_coef", 2.0))  # KL loss coefficient for individual policies
        self._team_kl_coef = self._as_dict(self.cfg.get("team_kl_coef", 1.0))  # KL loss coefficient for individual policies

        # set up automatic mixed precision
        self._device_type = torch.device(device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)

        # set up optimizer and learning rate scheduler
        self.idv_optimizers = {}
        self.team_optimizers = {}
        self.idv_schedulers = {}
        self.team_schedulers = {}

        for uid in self.possible_agents:
            # individual policy optimizer and scheduler
            idv_policy = self.idv_policies[uid]
            idv_value = self.idv_values[uid]
            if idv_policy is not None and idv_value is not None:
                if idv_policy is idv_value:
                    optimizer = torch.optim.Adam(idv_policy.parameters(), lr=self._learning_rate[uid])
                else:
                    optimizer = torch.optim.Adam(
                        itertools.chain(idv_policy.parameters(), idv_value.parameters()), lr=self._learning_rate[uid]
                    )
                self.idv_optimizers[uid] = optimizer
                if self._learning_rate_scheduler[uid] is not None:
                    self.idv_schedulers[uid] = self._learning_rate_scheduler[uid](
                        optimizer, **self._learning_rate_scheduler_kwargs[uid]
                    )
            
            # team policy optimizer and scheduler
            team_policy = self.team_policies[uid]
            team_value = self.team_values[uid]
            if team_policy is not None and team_value is not None:
                if team_policy is team_value:
                    optimizer = torch.optim.Adam(team_policy.parameters(), lr=self._learning_rate[uid])
                else:
                    optimizer = torch.optim.Adam(
                        itertools.chain(team_policy.parameters(), team_value.parameters()), lr=self._learning_rate[uid]
                    )
                self.team_optimizers[uid] = optimizer
                if self._learning_rate_scheduler[uid] is not None:
                    self.team_schedulers[uid] = self._learning_rate_scheduler[uid](
                        optimizer, **self._learning_rate_scheduler_kwargs[uid]
                    )

            self.checkpoint_modules[uid]["idv_optimizer"] = self.idv_optimizers[uid]

            # set up preprocessors
            if self._state_preprocessor[uid] is not None:
                self._state_preprocessor[uid] = self._state_preprocessor[uid](**self._state_preprocessor_kwargs[uid])
                self.checkpoint_modules[uid]["state_preprocessor"] = self._state_preprocessor[uid]
            else:
                self._state_preprocessor[uid] = self._empty_preprocessor

            if self._shared_state_preprocessor[uid] is not None:
                self._shared_state_preprocessor[uid] = self._shared_state_preprocessor[uid](
                    **self._shared_state_preprocessor_kwargs[uid]
                )
                self.checkpoint_modules[uid]["shared_state_preprocessor"] = self._shared_state_preprocessor[uid]
            else:
                self._shared_state_preprocessor[uid] = self._empty_preprocessor

            if self._value_preprocessor[uid] is not None:
                self._value_preprocessor[uid] = self._value_preprocessor[uid](**self._value_preprocessor_kwargs[uid])
                self.checkpoint_modules[uid]["value_preprocessor"] = self._value_preprocessor[uid]
            else:
                self._value_preprocessor[uid] = self._empty_preprocessor

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memories
        if self.memories:
            for uid in self.possible_agents:
                self.memories[uid].create_tensor(name="states", size=self.observation_spaces[uid], dtype=torch.float32)
                self.memories[uid].create_tensor(
                    name="shared_states", size=self.shared_observation_spaces[uid], dtype=torch.float32
                )
                self.memories[uid].create_tensor(name="actions", size=self.action_spaces[uid], dtype=torch.float32)
                self.memories[uid].create_tensor(name="idv_rewards", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="terminated", size=1, dtype=torch.bool)
                self.memories[uid].create_tensor(name="truncated", size=1, dtype=torch.bool)
                self.memories[uid].create_tensor(name="idv_log_prob", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="idv_values", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="idv_returns", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="idv_advantages", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="idv_dists_loc", size=self.action_spaces[uid], dtype=torch.float32)
                self.memories[uid].create_tensor(name="idv_dists_scale", size=self.action_spaces[uid], dtype=torch.float32)
                
                self.memories[uid].create_tensor(name="team_rewards", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="team_log_prob", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="team_values", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="team_returns", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="team_advantages", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="team_dists_loc", size=self.action_spaces[uid], dtype=torch.float32)
                self.memories[uid].create_tensor(name="team_dists_scale", size=self.action_spaces[uid], dtype=torch.float32)

                # tensors sampled during training
                self._tensors_names = [
                    "states",
                    "shared_states",
                    "actions",
                    "idv_log_prob",
                    "idv_values",
                    "idv_returns",
                    "idv_advantages",
                    "idv_dists_loc",
                    "idv_dists_scale",
                    "team_log_prob",
                    "team_values",
                    "team_returns",
                    "team_advantages",
                    "team_dists_loc",
                    "team_dists_scale",
                ]

        # create temporary variables needed for storage and computation
        self._current_team_log_prob = []
        self._current_idv_log_prob = []
        self._current_team_dists_loc = []
        self._current_team_dists_scale = []
        self._current_idv_dists_loc = []
        self._current_idv_dists_scale = []
        self._current_shared_next_states = []

    def act(self, states: Mapping[str, torch.Tensor], timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policies

        :param states: Environment's states
        :type states: dictionary of torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        # # sample random actions
        # # TODO: fix for stochasticity, rnn and log_prob
        # if timestep < self._random_timesteps:
        #     return self.policy.random_act({"states": states}, role="policy")

        # sample stochastic actions
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            idv_data = [
                self.idv_policies[uid].act({"states": self._state_preprocessor[uid](states[uid])}, role="policy")
                for uid in self.possible_agents
            ]
            
            idv_actions = {uid: d[0] for uid, d in zip(self.possible_agents, idv_data)}
            idv_log_prob = {uid: d[1] for uid, d in zip(self.possible_agents, idv_data)}
            idv_outputs = {uid: d[2] for uid, d in zip(self.possible_agents, idv_data)}
            idv_dists = {uid: out["distribution"] for uid, out in idv_outputs.items()}

            team_data = [
                self.team_policies[uid].act({"states": self._state_preprocessor[uid](states[uid])}, role="policy")
                for uid in self.possible_agents
            ]
            team_actions = {uid: d[0] for uid, d in zip(self.possible_agents, team_data)}
            team_log_prob = {uid: d[1] for uid, d in zip(self.possible_agents, team_data)}
            team_outputs = {uid: d[2] for uid, d in zip(self.possible_agents, team_data)}
            team_dists = {uid: out["distribution"] for uid, out in team_outputs.items()}

            if self._act_policy == "idv":
                actions = idv_actions
                team_log_prob = {}
                for uid in self.possible_agents:
                    state = self._state_preprocessor[uid](states[uid])
                    _, logp, _ = self.team_policies[uid].act(
                        {"states": state, "taken_actions": actions[uid]}, role="policy"
                    )
                    team_log_prob[uid] = logp
                
            elif self._act_policy == "team":
                # 추가: idv_policy 기준 log_prob 계산
                actions = team_actions
                idv_log_prob = {}
                for uid in self.possible_agents:
                    state = self._state_preprocessor[uid](states[uid])
                    _, logp, _ = self.idv_policies[uid].act(
                        {"states": state, "taken_actions": actions[uid]}, role="policy"
                    )
                    idv_log_prob[uid] = logp

            else:
                raise ValueError(f"Unknown act_policy: {self._act_policy}. Use 'idv' or 'team'.")

            self._current_idv_log_prob = idv_log_prob
            self._current_idv_dists_loc = {uid: dist.loc for uid, dist in idv_dists.items()}
            self._current_idv_dists_scale = {uid: dist.scale for uid, dist in idv_dists.items()}
            self._current_team_log_prob = team_log_prob
            self._current_team_dists_loc = {uid: dist.loc for uid, dist in team_dists.items()}
            self._current_team_dists_scale = {uid: dist.scale for uid, dist in team_dists.items()}

        if self._act_policy == "idv":
            log_prob, outputs = idv_log_prob, idv_outputs
        elif self._act_policy == "team":
            log_prob, outputs = team_log_prob, team_outputs
        return actions, log_prob, outputs

    def record_transition(
        self,
        states: Mapping[str, torch.Tensor],
        actions: Mapping[str, torch.Tensor],
        rewards: Mapping[str, torch.Tensor],
        next_states: Mapping[str, torch.Tensor],
        terminated: Mapping[str, torch.Tensor],
        truncated: Mapping[str, torch.Tensor],
        infos: Mapping[str, Any],
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: dictionary of torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: dictionary of torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: dictionary of torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: dictionary of torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: dictionary of torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: dictionary of torch.Tensor
        :param infos: Additional information about the environment
        :type infos: dictionary of any supported type
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super().record_transition(
            states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )

        if self.memories:
            shared_states = infos["shared_states"]
            self._current_shared_next_states = infos["shared_next_states"]

            for uid in self.possible_agents:
                team_uid = "team_" + uid

                # reward shaping
                if self._rewards_shaper is not None:
                    rewards[uid] = self._rewards_shaper(rewards[uid], timestep, timesteps)
                    rewards[team_uid] = self._rewards_shaper(rewards[team_uid], timestep, timesteps)

                # compute values
                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    idv_values, _, _ = self.idv_values[uid].act(
                        {"states": self._shared_state_preprocessor[uid](shared_states)}, role="value"
                    )
                    idv_values = self._value_preprocessor[uid](idv_values, inverse=True)
                    
                    team_values, _, _ = self.team_values[uid].act(
                        {"states": self._shared_state_preprocessor[uid](shared_states)}, role="value"
                    )
                    team_values = self._value_preprocessor[uid](team_values, inverse=True)

                # time-limit (truncation) bootstrapping
                if self._time_limit_bootstrap[uid]:
                    rewards[uid] += self._discount_factor[uid] * idv_values * truncated[uid]
                    rewards[team_uid] += self._discount_factor[uid] * team_values * truncated[uid]

                # storage transition in memory
                self.memories[uid].add_samples(
                    states=states[uid],
                    actions=actions[uid],
                    idv_rewards=rewards[uid],
                    team_rewards=rewards[team_uid],
                    next_states=next_states[uid],
                    terminated=terminated[uid],
                    truncated=truncated[uid],
                    idv_log_prob=self._current_idv_log_prob[uid],
                    team_log_prob=self._current_team_log_prob[uid],
                    idv_dists_loc=self._current_idv_dists_loc[uid],
                    idv_dists_scale=self._current_idv_dists_scale[uid],
                    team_dists_loc=self._current_team_dists_loc[uid],                    
                    team_dists_scale=self._current_team_dists_scale[uid],                    
                    idv_values=idv_values,
                    team_values=team_values,
                    shared_states=shared_states,
                )
                
    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """

        def compute_gae(
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            next_values: torch.Tensor,
            discount_factor: float = 0.99,
            lambda_coefficient: float = 0.95,
        ) -> torch.Tensor:
            """Compute the Generalized Advantage Estimator (GAE)

            :param rewards: Rewards obtained by the agent
            :type rewards: torch.Tensor
            :param dones: Signals to indicate that episodes have ended
            :type dones: torch.Tensor
            :param values: Values obtained by the agent
            :type values: torch.Tensor
            :param next_values: Next values obtained by the agent
            :type next_values: torch.Tensor
            :param discount_factor: Discount factor
            :type discount_factor: float
            :param lambda_coefficient: Lambda coefficient
            :type lambda_coefficient: float

            :return: Generalized Advantage Estimator
            :rtype: torch.Tensor
            """
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]

            # advantages computation
            for i in reversed(range(memory_size)):
                next_values = values[i + 1] if i < memory_size - 1 else idv_last_values
                advantage = (
                    rewards[i]
                    - values[i]
                    + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
                )
                advantages[i] = advantage
            # returns computation
            returns = advantages + values
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            return returns, advantages

        for uid in self.possible_agents:
            idv_policy = self.idv_policies[uid]
            idv_value = self.idv_values[uid]
            memory = self.memories[uid]
            
            team_policy = self.team_policies[uid]
            team_value = self.team_values[uid]

            # compute returns and advantages of individual policies
            with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                idv_value.train(False)
                idv_last_values, _, _ = idv_value.act(
                    {"states": self._shared_state_preprocessor[uid](self._current_shared_next_states.float())},
                    role="value",
                )
                idv_value.train(True)
            idv_last_values = self._value_preprocessor[uid](idv_last_values, inverse=True)

            idv_values = memory.get_tensor_by_name("idv_values")
            idv_returns, idv_advantages = compute_gae(
                rewards=memory.get_tensor_by_name("idv_rewards"),
                dones=memory.get_tensor_by_name("terminated") | memory.get_tensor_by_name("truncated"),
                values=idv_values,
                next_values=idv_last_values,
                discount_factor=self._discount_factor[uid],
                lambda_coefficient=self._lambda[uid],
            )

            memory.set_tensor_by_name("idv_values", self._value_preprocessor[uid](idv_values, train=True))
            memory.set_tensor_by_name("idv_returns", self._value_preprocessor[uid](idv_returns, train=True))
            memory.set_tensor_by_name("idv_advantages", idv_advantages)
            
            # compute returns and advantages of team policies
            with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                team_value.train(False)
                team_last_values, _, _ = team_value.act(
                    {"states": self._shared_state_preprocessor[uid](self._current_shared_next_states.float())},
                    role="value",
                )
                team_value.train(True)
            team_last_values = self._value_preprocessor[uid](team_last_values, inverse=True)
            
            team_values = memory.get_tensor_by_name("team_values")
            team_returns, team_advantages = compute_gae(
                rewards=memory.get_tensor_by_name("team_rewards"),
                dones=memory.get_tensor_by_name("terminated") | memory.get_tensor_by_name("truncated"),
                values=team_values,
                next_values=team_last_values,
                discount_factor=self._discount_factor[uid],
                lambda_coefficient=self._lambda[uid],
            )
            memory.set_tensor_by_name("team_values", self._value_preprocessor[uid](team_values, train=True))
            memory.set_tensor_by_name("team_returns", self._value_preprocessor[uid](team_returns, train=True))
            memory.set_tensor_by_name("team_advantages", team_advantages)

            # sample mini-batches from memory
            sampled_batches = memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches[uid])

            idv_cumulative_policy_loss = 0
            idv_cumulative_entropy_loss = 0
            idv_cumulative_value_loss = 0

            team_cumulative_policy_loss = 0
            team_cumulative_entropy_loss = 0
            team_cumulative_value_loss = 0
            
            # learning epochs
            for epoch in range(self._learning_epochs[uid]):
                idv_kl_divergences = []
                team_kl_divergences = []

                # mini-batches loop of individual agents
                for (
                    sampled_states,
                    sampled_shared_states,
                    sampled_actions,
                    idv_sampled_log_prob,
                    idv_sampled_values,
                    idv_sampled_returns,
                    idv_sampled_advantages,
                    idv_sampled_dists_loc,
                    idv_sampled_dists_scale,
                    team_sampled_log_prob,
                    team_sampled_values,
                    team_sampled_returns,
                    team_sampled_advantages,                    
                    team_sampled_dists_loc,                    
                    team_sampled_dists_scale,                    
                ) in sampled_batches:

                    with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                        # set the old distribution parameters
                        idv_sampled_old_dists = torch.distributions.Normal(
                                loc=idv_sampled_dists_loc.clone().detach(),
                                scale=idv_sampled_dists_scale.clone().detach(),
                            )
                        
                        team_sampled_old_dists = torch.distributions.Normal(
                                loc=team_sampled_dists_loc.clone().detach(),
                                scale=team_sampled_dists_scale.clone().detach(),
                            )
                        
                        # preprocess states
                        sampled_states = self._state_preprocessor[uid](sampled_states, train=not epoch)
                        sampled_shared_states = self._shared_state_preprocessor[uid](
                            sampled_shared_states, train=not epoch
                        )
                        
                        # clamp old logprobs for stability
                        idv_sampled_log_prob = torch.clamp(idv_sampled_log_prob, min=-20, max=2)
                        team_sampled_log_prob = torch.clamp(team_sampled_log_prob, min=-20, max=2)
                        
                        # evaluate current policies                        

                        _, idv_next_log_prob, idv_next_outputs = idv_policy.act(
                            {"states": sampled_states, "taken_actions": sampled_actions}, role="policy"
                        )
                        idv_next_dists = idv_next_outputs["distribution"]
                        idv_next_entropy = idv_policy.get_entropy(role="policy").mean()
                        
                        # compute approximate KL divergence
                        with torch.no_grad():
                            ratio = idv_next_log_prob - idv_sampled_log_prob
                            kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                            idv_kl_divergences.append(kl_divergence)
                        
                        _, team_next_log_prob, team_next_outputs = team_policy.act(
                            {"states": sampled_states, "taken_actions": sampled_actions}, role="policy"
                        )
                        
                        # compute approximate KL divergence
                        with torch.no_grad():
                            ratio = team_next_log_prob - team_sampled_log_prob
                            kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                            team_kl_divergences.append(kl_divergence)
                        
                        team_next_dists = team_next_outputs["distribution"]
                        team_next_entropy = team_policy.get_entropy(role="policy").mean()
                        
                        idv_predicted_values, _, _ = idv_value.act({"states": sampled_shared_states}, role="value")
                        team_predicted_values, _, _ = team_value.act({"states": sampled_shared_states}, role="value")

                        # --- KL & Cross-Entropy Loss ---
                        idv_kl_loss = torch.tensor(0.0, device=self.device)
                        team_kl_loss = torch.tensor(0.0, device=self.device)
                        idv_cross_entropy = torch.tensor(0.0, device=self.device)
                        team_cross_entropy = torch.tensor(0.0, device=self.device)
                        
                        # compute KL divergence and cross-entropy loss between old team and new individual policies
                        idv_kl_loss += torch.distributions.kl_divergence(team_sampled_old_dists, idv_next_dists).mean()
                        if self._idv_use_cross_entropy:
                            sample_from_team = team_sampled_old_dists.rsample()
                            idv_cross_entropy -= idv_next_dists.log_prob(sample_from_team).sum(dim=-1).mean()

                        # compute KL divergence and cross-entropy loss between old individual and new team policies
                        team_kl_loss += torch.distributions.kl_divergence(idv_sampled_old_dists, team_next_dists).mean()
                        if self._team_use_cross_entropy:
                            sample_from_idv = idv_sampled_old_dists.rsample()
                            team_cross_entropy -= team_next_dists.log_prob(sample_from_idv).sum(dim=-1).mean()

                        idv_imp_weights = torch.exp(idv_next_log_prob - idv_sampled_log_prob)
                        team_imp_weights = torch.exp(team_next_log_prob - team_sampled_log_prob)

                        if self._idv_clip_use_present:
                            so_weights = torch.exp(idv_next_log_prob - team_next_log_prob.clone().detach())
                        else:
                            so_weights = torch.exp(idv_next_log_prob - team_sampled_log_prob.clone().detach())

                        idv_surr1 = idv_sampled_advantages * idv_imp_weights
                        idv_surr2 = torch.clamp(
                            idv_imp_weights, 1.0 - self._ratio_clip[uid], 1.0 + self._ratio_clip[uid]
                        ) * idv_sampled_advantages
                        idv_clp = torch.clamp(so_weights, 1.0 - self._idv_ratio_clip[uid],
                                              1.0 + self._idv_ratio_clip[uid]) 
                        idv_surr3 = idv_sampled_advantages * idv_clp
                        
                        idv_min = torch.min(idv_surr1, idv_surr2)
                        if self._idv_use_two_clip:
                            idv_min = torch.min(idv_min, idv_surr3)
                            
                        policy_action_loss = (-1) * idv_min.mean()
                        idv_policy_loss = policy_action_loss
                        idv_loss = idv_policy_loss - idv_next_entropy * self._entropy_loss_scale[uid]
                        
                        if self._idv_use_kl_loss:
                            idv_loss += self._idv_kl_coef[uid] * idv_kl_loss
                        elif self._idv_use_cross_entropy:
                            idv_loss += self._idv_kl_coef[uid] * idv_cross_entropy
                        else:
                            idv_kl_loss = torch.tensor(0.0, device=self.device)

                        # compute value loss
                        idv_predicted_values, _, _ = idv_value.act({"states": sampled_shared_states}, role="value")

                        if self._clip_predicted_values:
                            idv_predicted_values = idv_sampled_values + torch.clip(
                                idv_predicted_values - idv_sampled_values, min=-self._value_clip[uid], max=self._value_clip[uid]
                            )
                        idv_value_loss = self._value_loss_scale[uid] * F.mse_loss(idv_sampled_returns, idv_predicted_values)

                        if self._team_clip_use_present:
                            team_imp_weights = torch.exp(team_next_log_prob - idv_next_log_prob.clone().detach())
                        else:
                            team_imp_weights = torch.exp(team_next_log_prob - idv_sampled_log_prob.clone().detach())

                        team_surr1 = team_sampled_advantages * team_imp_weights
                        tclp = torch.clamp(team_imp_weights, 1.0 - self._ratio_clip[uid], 1.0 + self._ratio_clip[uid])
                        team_surr2 = tclp * team_sampled_advantages
                        
                        team_min = team_surr1
                        if self._team_use_clip:
                            team_min = torch.min(team_surr1, team_surr2)
                        policy_action_loss = (-1) * team_min.mean()
                        team_policy_loss = policy_action_loss
                        team_loss = policy_action_loss - team_next_entropy * self._entropy_loss_scale[uid]
                        
                        if self._team_use_kl_loss:
                            team_loss += self._team_kl_coef[uid] * team_kl_loss
                        elif self._team_use_cross_entropy:
                            team_loss += self._team_kl_coef[uid] * team_cross_entropy
                        else:
                            team_kl_loss = torch.tensor(0.0, device=self.device)
                            
                        # compute TEAM value loss
                        team_predicted_values, _, _ = team_value.act({"states": sampled_shared_states}, role="value")

                        if self._clip_predicted_values:
                            team_predicted_values = team_sampled_values + torch.clip(
                                team_predicted_values - team_sampled_values, min=-self._value_clip[uid], max=self._value_clip[uid]
                            )
                        team_value_loss = self._value_loss_scale[uid] * F.mse_loss(team_sampled_returns, team_predicted_values)

                    # update individual actor
                    if torch.isnan(idv_loss).any():
                        print("idv loss has nan")
                    if torch.isinf(idv_loss).any():
                        print("idv loss has inf")
                    if torch.isnan(so_weights).any():
                        print("so_weights has nan")
                    if torch.isinf(so_weights).any():
                        print("so_weights has inf")
                    if torch.isnan(idv_imp_weights).any():
                        print("idv imp_weights has nan")
                    if torch.isinf(idv_imp_weights).any():
                        print("idv imp_weights has inf")
                    # update team actor
                    if torch.isnan(team_loss).any():
                        print("team loss has nan")
                    if torch.isinf(team_loss).any():
                        print("team loss has inf")
                    if torch.isnan(team_imp_weights).any():
                        print("team has nan")
                    if torch.isinf(team_imp_weights).any():
                        print("team has inf")

                    # optimization individual step
                    self.idv_optimizers[uid].zero_grad()
                    self.scaler.scale(idv_loss + idv_value_loss).backward()

                    if config.torch.is_distributed:
                        idv_policy.reduce_parameters()
                        if idv_policy is not idv_value:
                            idv_value.reduce_parameters()

                    if self._grad_norm_clip[uid] > 0:
                        self.scaler.unscale_(self.idv_optimizers[uid])
                        if idv_policy is idv_value:
                            nn.utils.clip_grad_norm_(idv_policy.parameters(), self._grad_norm_clip[uid])
                        else:
                            nn.utils.clip_grad_norm_(
                                itertools.chain(idv_policy.parameters(), idv_value.parameters()), self._grad_norm_clip[uid]
                            )

                    self.scaler.step(self.idv_optimizers[uid])
                    self.scaler.update()

                    # update cumulative losses
                    idv_cumulative_policy_loss += idv_policy_loss.item()
                    idv_cumulative_value_loss += idv_value_loss.item()
                    if self._entropy_loss_scale[uid]:
                        if self._idv_use_kl_loss:
                            idv_cumulative_entropy_loss += idv_kl_loss.item()
                        elif self._idv_use_cross_entropy:
                            idv_cumulative_entropy_loss += idv_cross_entropy.item()
                        else:
                            idv_cumulative_entropy_loss += 0.0
                            
                    # optimization team step
                    self.team_optimizers[uid].zero_grad()
                    self.scaler.scale(team_loss + team_value_loss).backward()

                    if config.torch.is_distributed:
                        team_policy.reduce_parameters()
                        if team_policy is not team_value:
                            team_value.reduce_parameters()

                    if self._grad_norm_clip[uid] > 0:
                        self.scaler.unscale_(self.team_optimizers[uid])
                        if team_policy is team_value:
                            nn.utils.clip_grad_norm_(team_policy.parameters(), self._grad_norm_clip[uid])
                        else:
                            nn.utils.clip_grad_norm_(
                                itertools.chain(team_policy.parameters(), team_value.parameters()), self._grad_norm_clip[uid]
                            )

                    self.scaler.step(self.team_optimizers[uid])
                    self.scaler.update()

                    # update cumulative losses
                    team_cumulative_policy_loss += team_policy_loss.item()
                    team_cumulative_value_loss += team_value_loss.item()
                    if self._entropy_loss_scale[uid]:
                        if self._team_use_kl_loss:
                            team_cumulative_entropy_loss += team_kl_loss.item()
                        elif self._idv_use_cross_entropy:
                            team_cumulative_entropy_loss += team_cross_entropy.item()
                        else:
                            team_cumulative_entropy_loss += 0.0

                # update individual policy learning rate
                if self._learning_rate_scheduler[uid]:
                    if isinstance(self.idv_schedulers[uid], KLAdaptiveLR):
                        kl = torch.tensor(idv_kl_divergences, device=self.device).mean()
                        # reduce (collect from all workers/processes) KL in distributed runs
                        if config.torch.is_distributed:
                            torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                            kl /= config.torch.world_size
                        self.idv_schedulers[uid].step(kl.item())
                    else:
                        self.idv_schedulers[uid].step()
                        
                # update team policy learning rate
                if self._learning_rate_scheduler[uid]:
                    if isinstance(self.team_schedulers[uid], KLAdaptiveLR):
                        kl = torch.tensor(team_kl_divergences, device=self.device).mean()
                        # reduce (collect from all workers/processes) KL in distributed runs
                        if config.torch.is_distributed:
                            torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                            kl /= config.torch.world_size
                        self.team_schedulers[uid].step(kl.item())
                    else:
                        self.team_schedulers[uid].step()

            # record individual data
            self.track_data(
                f"Loss / individual Policy loss ({uid})",
                idv_cumulative_policy_loss / (self._learning_epochs[uid] * self._mini_batches[uid]),
            )
            self.track_data(
                f"Loss / individual Value loss ({uid})",
                idv_cumulative_value_loss / (self._learning_epochs[uid] * self._mini_batches[uid]),
            )
            if self._entropy_loss_scale:
                self.track_data(
                    f"Loss / individual Entropy loss ({uid})",
                    idv_cumulative_entropy_loss / (self._learning_epochs[uid] * self._mini_batches[uid]),
                )

            self.track_data(
                f"Policy / individual Standard deviation ({uid})", idv_policy.distribution(role="policy").stddev.mean().item()
            )

            if self._learning_rate_scheduler[uid]:
                self.track_data(f"Learning / individual Learning rate individual ({uid})", self.idv_schedulers[uid].get_last_lr()[0])

            # record team data
            self.track_data(
                f"Loss / team Policy loss ({uid})",
                team_cumulative_policy_loss / (self._learning_epochs[uid] * self._mini_batches[uid]),
            )
            self.track_data(
                f"Loss / team Value loss ({uid})",
                team_cumulative_value_loss / (self._learning_epochs[uid] * self._mini_batches[uid]),
            )
            if self._entropy_loss_scale:
                self.track_data(
                    f"Loss / team Entropy loss ({uid})",
                    team_cumulative_entropy_loss / (self._learning_epochs[uid] * self._mini_batches[uid]),
                )

            self.track_data(
                f"Policy / team Standard deviation ({uid})", team_policy.distribution(role="policy").stddev.mean().item()
            )

            if self._learning_rate_scheduler[uid]:
                self.track_data(f"Learning / team Learning rate individual ({uid})", self.team_schedulers[uid].get_last_lr()[0])
