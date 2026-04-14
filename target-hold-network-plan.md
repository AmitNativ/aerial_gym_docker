# Custom Asymmetric PPO Network for target_hold

## Architecture overview

Asymmetric actor-critic with GRU memory on the actor only.

```
              obs (22 dims)
              ┌─────────────────────────────────────────────┐
              │ [0:12] actor  [12:16] prev_act  [16:22] priv│
              └──┬──────────────┬────────────────────┬──────┘
                 │              │                    │
        ┌───────▼───────┐      │         ┌──────────▼──────────┐
        │ Actor Encoder │      │         │   Critic MLP        │
        │ FC(12→128,ELU)│      │         │   FC(22→128,ELU)    │
        │ FC(128→128,ELU│      │         │   FC(128→128,ELU)   │
        └───────┬───────┘      │         └──────────┬──────────┘
                │ (128)        │ (4)                │ (128)
                └──────┬───────┘                    │
                       │ concat (132)               │
                ┌──────▼──────┐              ┌──────▼──────┐
                │  GRU cell   │              │   Linear    │
                │  hidden=64  │              │  128 → 1    │
                └──────┬──────┘              └──────┬──────┘
                       │ (64)                       │ (1)
                ┌──────▼──────┐                     │
                │   Linear    │                     │
                │   64 → 4    │                     │
                └──────┬──────┘                     │
                       │                            │
                    mu (4)    sigma (4, learned)   value
```

## Observation tensor layout (22 dims)

Packed into a single flat tensor (rl_games constraint).

| Indices | Dims | Source | Consumer |
|---|---|---|---|
| `[0:4]` | 4 | `robot_orientation` quaternion (qw>0 enforced) | Actor encoder |
| `[4:7]` | 3 | `robot_body_angvel` (body frame, rad/s) | Actor encoder |
| `[7:11]` | 4 | target bbox [tl_x, tl_y, br_x, br_y] normalized [-1,1] | Actor encoder |
| `[11:12]` | 1 | vz — vertical velocity (world frame, m/s) | Actor encoder |
| `[12:16]` | 4 | previous actions (raw [-1,1] policy outputs) | GRU input (concat with encoder output) |
| `[16:19]` | 3 | robot position (world frame) | Critic only (privileged) |
| `[19:22]` | 3 | robot linear velocity vx,vy,vz (world frame) | Critic only (privileged) |

- **Actor** reads `[0:12]` for the encoder, `[12:16]` for GRU prev_actions.
- **Critic** reads `[0:22]` (everything including privileged state).

## Actor detail

1. **Encoder**: `obs[0:12]` → FC(12→128, ELU) → FC(128→128, ELU) → 128-dim encoding
2. **GRU input**: `concat(encoding, obs[12:16])` → 132-dim vector
3. **GRU cell**: `GRUWithDones(input=132, hidden=64, layers=1)` — handles episode resets via done flags automatically
4. **Action head**: `Linear(64→4)` → mu (action mean)
5. **Sigma**: learnable `nn.Parameter(torch.zeros(4))` — fixed log-std, not state-dependent

## Critic detail

1. **MLP**: `obs[0:22]` → FC(22→128, ELU) → FC(128→128, ELU) → 128-dim
2. **Value head**: `Linear(128→1)` → scalar value
3. No RNN — critic is a pure feedforward network

## Action space

4 dims, policy outputs in [-1, 1], scaled by `action_transformation_function`:

| Index | Meaning | Scaling |
|---|---|---|
| `[0]` | thrust | `(x+1)/2 * max_thrust_m_s2` → [0, max] m/s² |
| `[1]` | roll rate | `x * max_roll_rate_rad_s` → [-max, max] rad/s |
| `[2]` | pitch rate | `x * max_pitch_rate_rad_s` → [-max, max] rad/s |
| `[3]` | yaw rate | `x * max_yaw_rate_rad_s` → [-max, max] rad/s |

Controller: `lee_rates_control` (expects [thrust, roll_rate, pitch_rate, yaw_rate]).

## rl_games integration

### Registration

```python
from rl_games.algos_torch import model_builder
from target_hold.networks.target_hold_network import TargetHoldNetworkBuilder
model_builder.register_network('target_hold_actor_critic', TargetHoldNetworkBuilder)
```

Done in `target_hold/training/runner.py` before the rl_games Runner is created.

### YAML config (network section)

```yaml
network:
  name: target_hold_actor_critic
  actor_obs_dim: 12
  prev_actions_dim: 4
  critic_obs_dim: 22
  actor:
    encoder_units: [128, 128]
    activation: elu
  critic:
    units: [128, 128]
    activation: elu
  rnn:
    units: 64
    layers: 1
  space:
    continuous:
      mu_activation: None
      sigma_activation: None
      mu_init:
        name: default
      sigma_init:
        name: const_initializer
        val: 0
      fixed_sigma: True
```

### Key PPO config for RNN

```yaml
config:
  seq_length: 16        # BPTT sequence length for GRU training
  horizon_length: 64    # rollout length (must be divisible by seq_length)
```

### rl_games API contracts

- `forward()` returns `(mu, sigma, value, states)` for continuous actions
- `states` is always a tuple — for our single GRU: `(hidden_state,)`
- `is_rnn()` returns `True` — tells the agent to manage hidden states and use `seq_length`
- `is_separate_critic()` returns `True` — actor and critic have independent paths
- `get_default_rnn_state()` returns `(torch.zeros(num_layers, num_seqs, rnn_units),)`
- `GRUWithDones` automatically zeros hidden states on episode boundaries via done flags
- The agent also zeros RNN states for done envs between steps (`zero_rnn_on_done`)

## Files

| File | Role |
|---|---|
| `target_hold/networks/target_hold_network.py` | Network definition (`TargetHoldNetworkBuilder`) |
| `target_hold/networks/__init__.py` | Package marker |
| `target_hold/training/runner.py` | Registers network before rl_games Runner |
| `target_hold/training/ppo_target_hold.yaml` | PPO config referencing `target_hold_actor_critic` |
| `target_hold/configs/target_hold_task_config.py` | `observation_space_dim=22`, action scaling params |
| `target_hold/tasks/target_hold_task.py` | Packs 22-dim obs, zeros prev_actions on reset |
