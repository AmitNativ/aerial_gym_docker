"""Asymmetric actor-critic network for the target-hold task.

Actor:  obs[0:12] → FC(128,ELU) → FC(128,ELU) → concat(enc, prev_actions) → GRU(64) → Linear → 4-dim mu
Critic: obs[0:22] → FC(128,ELU) → FC(128,ELU) → Linear → 1-dim value

Observation layout (22 dims packed by the task):
  [0:12]  actor obs (quat, body angvel, bbox, vz)
  [12:16] previous actions (raw [-1,1] policy outputs)
  [16:22] privileged (world pos, world linvel)

Sigma is a learnable parameter (fixed log-std, not state-dependent).
"""

import torch
from torch import nn

from rl_games.algos_torch.network_builder import NetworkBuilder
from rl_games.common.layers.recurrent import GRUWithDones


class TargetHoldNetworkBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return TargetHoldNetworkBuilder.Network(self.params, **kwargs)

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            NetworkBuilder.BaseNetwork.__init__(self)

            self.actions_num = kwargs.pop("actions_num")
            self.input_shape = kwargs.pop("input_shape")
            self.value_size = kwargs.pop("value_size", 1)
            self.num_seqs = kwargs.pop("num_seqs", 1)

            # Read dims from YAML params (with defaults matching our task config).
            self.actor_obs_dim = params.get("actor_obs_dim", 12)
            self.prev_actions_dim = params.get("prev_actions_dim", 4)
            self.critic_obs_dim = params.get("critic_obs_dim", 22)
            rnn_cfg = params.get("rnn", {})
            self.rnn_units = rnn_cfg.get("units", 64)
            self.rnn_layers = rnn_cfg.get("layers", 1)

            actor_cfg = params.get("actor", {})
            actor_units = actor_cfg.get("encoder_units", [128, 128])
            critic_cfg = params.get("critic", {})
            critic_units = critic_cfg.get("units", [128, 128])

            # ---- Actor ----
            # Encoder: obs[0:12] → FC → FC
            actor_layers = []
            in_dim = self.actor_obs_dim
            for units in actor_units:
                actor_layers.append(nn.Linear(in_dim, units))
                actor_layers.append(nn.ELU())
                in_dim = units
            self.actor_encoder = nn.Sequential(*actor_layers)

            # GRU: input = encoder_out + prev_actions
            gru_input_dim = in_dim + self.prev_actions_dim
            self.actor_gru = GRUWithDones(
                input_size=gru_input_dim,
                hidden_size=self.rnn_units,
                num_layers=self.rnn_layers,
            )

            # Action head
            self.mu = nn.Linear(self.rnn_units, self.actions_num)

            # Learnable log-std (fixed sigma — not state-dependent)
            self.sigma = nn.Parameter(
                torch.zeros(self.actions_num, requires_grad=True, dtype=torch.float32),
                requires_grad=True,
            )

            # ---- Critic ----
            # MLP: obs[0:22] → FC → FC → value
            critic_layers = []
            in_dim = self.critic_obs_dim
            for units in critic_units:
                critic_layers.append(nn.Linear(in_dim, units))
                critic_layers.append(nn.ELU())
                in_dim = units
            self.critic_mlp = nn.Sequential(*critic_layers)
            self.value = nn.Linear(in_dim, self.value_size)

            # Weight init (rl_games default style).
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            # Smaller init for action head for near-zero initial outputs.
            nn.init.orthogonal_(self.mu.weight, gain=0.01)

        # ----------------------------------------------------------
        # rl_games BaseNetwork API
        # ----------------------------------------------------------

        def is_rnn(self):
            return True

        def is_separate_critic(self):
            return True

        def get_default_rnn_state(self):
            # Only the actor has a GRU — single hidden state tensor.
            return (
                torch.zeros(
                    (self.rnn_layers, self.num_seqs, self.rnn_units),
                    dtype=torch.float32,
                ),
            )

        def forward(self, obs_dict):
            obs = obs_dict["obs"]  # (batch, 22)
            states = obs_dict.get("rnn_states", None)
            dones = obs_dict.get("dones", None)
            bptt_len = obs_dict.get("bptt_len", 0)
            seq_length = obs_dict.get("seq_length", 1)

            # Slice observation channels.
            actor_obs = obs[:, : self.actor_obs_dim]  # (batch, 12)
            prev_actions = obs[
                :, self.actor_obs_dim : self.actor_obs_dim + self.prev_actions_dim
            ]  # (batch, 4)

            # ---- Actor path ----
            enc = self.actor_encoder(actor_obs)  # (batch, 128)
            gru_in = torch.cat([enc, prev_actions], dim=-1)  # (batch, 132)

            # Reshape for GRU: (batch,) → (seq_len, num_seqs, features)
            batch_size = gru_in.size(0)
            num_seqs = batch_size // seq_length
            gru_in = gru_in.reshape(num_seqs, seq_length, -1).transpose(0, 1)

            if dones is not None:
                dones = dones.reshape(num_seqs, seq_length, -1).transpose(0, 1)

            # Unpack hidden state (single-element tuple).
            if states is not None and len(states) == 1:
                h = states[0]
            else:
                h = states

            gru_out, new_h = self.actor_gru(gru_in, h, dones, bptt_len)

            # Reshape back: (seq_len, num_seqs, rnn_units) → (batch, rnn_units)
            gru_out = gru_out.transpose(0, 1).contiguous().reshape(batch_size, -1)

            mu = self.mu(gru_out)  # (batch, 4)
            sigma = self.sigma.unsqueeze(0).expand_as(mu)  # (batch, 4)

            # ---- Critic path (no RNN) ----
            c_out = self.critic_mlp(obs[:, : self.critic_obs_dim])  # (batch, 128)
            value = self.value(c_out)  # (batch, 1)

            # Pack hidden state back into a tuple.
            if not isinstance(new_h, tuple):
                new_h = (new_h,)

            return mu, sigma, value, new_h
