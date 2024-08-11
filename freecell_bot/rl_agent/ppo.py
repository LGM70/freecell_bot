"""Adapted from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py"""
from ast import arg
import os
import random
import time
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim
from torch.distributions.categorical import Categorical
import tyro
from tqdm import tqdm
import wandb

# pylint: disable=E0402
from ..game_env.freecell_env import FreeCellEnv
from ..game_env.obs_space.compact_obs import CompactObsSpace as ObsSpace
from ..game_env.action_space.tuple_action import TupleActionSpace as ActionSpace
from .networks import FullyConnectedNetwork as Network

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    wandb_project_name: str = "freecell_bot"
    """the wandb's project name"""

    # Algorithm specific arguments
    total_timesteps: int = int(1e6)
    """total timesteps of the experiments"""
    learning_rate: float = 1e-3
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 32
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env():
    def thunk():
        env = FreeCellEnv(ObsSpace, ActionSpace)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


class Agent(nn.Module):
    def __init__(self, envs: gym.vector.VectorEnv):
        super().__init__()
        self.critic = Network(envs.single_observation_space.shape, 1)
        self.actor1 = Network(envs.single_observation_space.shape, envs.single_action_space.nvec[0])
        self.actor2 = Network(envs.single_observation_space.shape, envs.single_action_space.nvec[1])

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action: torch.Tensor = None):
        logits1, logits2 = self.actor1(x), self.actor2(x)
        overall_logits = (logits1.unsqueeze(-1) * logits2.unsqueeze(1)).flatten(start_dim=1)
        probs = Categorical(logits=overall_logits)
        if action is None:
            encoded_action = probs.sample()
            action = torch.stack([encoded_action // logits2.shape[-1], encoded_action % logits2.shape[-1]], dim=1)
        else:
            encoded_action = action[:, 0] * logits2.shape[-1] + action[:, 1]
        return action, probs.log_prob(encoded_action), probs.entropy(), self.critic(x)


if __name__ == '__main__':
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f'{args.exp_name}__{args.seed}__{int(time.time())}'

    wandb.require("core")
    wandb.init(
        project=args.wandb_project_name,
        config=vars(args),
        name=run_name,
        save_code=True,
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env() for _ in range(args.num_envs)],
    )

    if envs.single_action_space.sample().shape != (2,):
        raise NotImplementedError('this module only supports tuple action space')

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    with tqdm(total=args.num_iterations * args.num_steps * args.num_envs) as pbar:
        for iteration in range(1, args.num_iterations + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]['lr'] = lrnow

            for step in range(0, args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward, dtype=torch.float).to(device).view(-1)
                next_obs = torch.tensor(next_obs, dtype=torch.float).to(device)
                next_done = torch.tensor(next_done, dtype=torch.float).to(device)

                if 'final_info' in infos:
                    for info in infos['final_info']:
                        if info and 'episode' in info:
                            wandb.log({
                                'charts/episodic_return': info['episode']['r'],
                                'charts/episodic_length': info['episode']['l'],
                            }, global_step)

                pbar.update(args.num_envs)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            wandb.log({
                'charts/learning_rate': optimizer.param_groups[0]['lr'],
                'charts/SPS': int(global_step / (time.time() - start_time)),
                'losses/value_loss': v_loss.item(),
                'losses/policy_loss': pg_loss.item(),
                'losses/entropy': entropy_loss.item(),
                'losses/old_approx_kl': old_approx_kl.item(),
                'losses/approx_kl': approx_kl.item(),
                'losses/clipfrac': np.mean(clipfracs),
                'losses/explained_variance': explained_var,
            }, global_step)

    envs.close()
