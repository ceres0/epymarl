# Modified from EPyMARL for OA-QMIX and opponent-profile research.
import copy

import torch as th
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from modules.critics.opponent_aux import OpponentAuxHead, masked_aux_loss
from modules.mixers.oa_qmix import OAQMixer
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer


class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                assert args.common_reward, "VDN only supports common reward setting"
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                assert args.common_reward, "QMIX only supports common reward setting"
                self.mixer = OAQMixer(args) if getattr(args, "oa_qmix_enabled", False) else QMixer(args)
            elif args.mixer == "oa_qmix":
                assert args.common_reward, "OAQMIX only supports common reward setting"
                self.mixer = OAQMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.aux_head = None
        if getattr(args, "opponent_aux_enabled", False):
            input_dim = int(args.hidden_dim) + int(getattr(args, "opp_profile_dim", 0) or 0)
            self.aux_head = OpponentAuxHead(
                input_dim=input_dim,
                hidden_dim=args.mixing_embed_dim,
                style_classes=4,
                n_agents=args.n_agents,
            )
            self.params += list(self.aux_head.parameters())

        self.optimiser = Adam(params=self.params, lr=args.lr)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.training_steps = 0
        self.last_target_update_step = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            rew_shape = (1,) if self.args.common_reward else (self.n_agents,)
            self.rew_ms = RunningMeanStd(shape=rew_shape, device=device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        if self.args.common_reward:
            assert (
                rewards.size(2) == 1
            ), "Expected singular agent dimension for common rewards"
            # reshape rewards to be of shape (batch_size, episode_length, n_agents)
            rewards = rewards.expand(-1, -1, self.n_agents)

        # Calculate estimated Q-Values
        mac_out = []
        mac_hidden = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            if self.aux_head is not None:
                mac_hidden.append(
                    self.mac.hidden_states.reshape(batch.batch_size, self.n_agents, -1)
                )
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        if mac_hidden:
            mac_hidden = th.stack(mac_hidden, dim=1)
        else:
            mac_hidden = None
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(
            3
        )  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            if isinstance(self.mixer, OAQMixer):
                chosen_action_qvals = self.mixer(
                    chosen_action_qvals,
                    batch["state"][:, :-1],
                    batch["opp_profile"][:, :-1] if "opp_profile" in batch.scheme else None,
                )
                target_max_qvals = self.target_mixer(
                    target_max_qvals,
                    batch["state"][:, 1:],
                    batch["opp_profile"][:, 1:] if "opp_profile" in batch.scheme else None,
                )
            else:
                chosen_action_qvals = self.mixer(
                    chosen_action_qvals, batch["state"][:, :-1]
                )
                target_max_qvals = self.target_mixer(
                    target_max_qvals, batch["state"][:, 1:]
                )

        if self.args.standardise_returns:
            target_max_qvals = (
                target_max_qvals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean
            )

        # Calculate 1-step Q-Learning targets
        targets = (
            rewards + self.args.gamma * (1 - terminated) * target_max_qvals.detach()
        )

        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        # Td-error
        td_error = chosen_action_qvals - targets.detach()

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error**2).sum() / mask.sum()
        aux_metrics = None
        if self.aux_head is not None and "opp_label_mask" in batch.scheme:
            aux_loss, aux_metrics = self._opponent_aux_loss(batch, mask, mac_hidden)
            loss = loss + self.args.opponent_aux_loss_weight * aux_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.training_steps += 1
        if (
            self.args.target_update_interval_or_tau > 1
            and (self.training_steps - self.last_target_update_step)
            / self.args.target_update_interval_or_tau
            >= 1.0
        ):
            self._update_targets_hard()
            self.last_target_update_step = self.training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            if isinstance(self.mixer, OAQMixer):
                self.logger.log_stat("oa_qmix_enabled", 1.0, t_env)
                self.logger.log_stat("critic_input_dim", self.mixer.critic_input_dim, t_env)
                if "opp_profile" in batch.scheme:
                    self.logger.log_stat(
                        "opp_profile_norm",
                        batch["opp_profile"][:, :-1].norm(dim=-1).mean().item(),
                        t_env,
                    )
            if aux_metrics is not None:
                for key, value in aux_metrics.items():
                    self.logger.log_stat("aux/" + key, value, t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat(
                "td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env
            )
            self.logger.log_stat(
                "q_taken_mean",
                (chosen_action_qvals * mask).sum().item()
                / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.logger.log_stat(
                "target_mean",
                (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.log_stats_t = t_env

    def _opponent_aux_loss(self, batch, mask, mac_hidden):
        if mac_hidden is None:
            raise RuntimeError("opponent auxiliary loss requires live MAC hidden states")

        # Use the live agent representation so the auxiliary objective trains
        # parameters that also affect action values, not only a detached side head.
        agent_repr = mac_hidden[:, :-1].mean(dim=2).reshape(-1, self.args.hidden_dim)
        opp_profile = batch["opp_profile"][:, :-1].reshape(-1, int(getattr(self.args, "opp_profile_dim", 0) or 0))
        label_mask = batch["opp_label_mask"][:, :-1].reshape(-1, 3) > 0
        labels = {
            "style": batch["opp_style_label"][:, :-1].reshape(-1),
            "focus": batch["enemy_focus_target"][:, :-1].reshape(-1),
            "danger": batch["danger_window"][:, :-1].reshape(-1),
        }
        valid = mask.reshape(-1, mask.size(-1)).any(dim=-1)
        masks = {
            "style": label_mask[:, 0] & valid,
            "focus": label_mask[:, 1] & valid,
            "danger": label_mask[:, 2] & valid,
        }
        outputs = self.aux_head(agent_repr, opp_profile)
        return masked_aux_loss(outputs, labels, masks)

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(
            self.target_mac.parameters(), self.mac.parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        if self.mixer is not None:
            for target_param, param in zip(
                self.target_mixer.parameters(), self.mixer.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        if self.aux_head is not None:
            self.aux_head.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        if self.aux_head is not None:
            th.save(self.aux_head.state_dict(), "{}/opponent_aux.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                th.load(
                    "{}/mixer.th".format(path),
                    map_location=lambda storage, loc: storage,
                )
            )
        if self.aux_head is not None:
            self.aux_head.load_state_dict(
                th.load(
                    "{}/opponent_aux.th".format(path),
                    map_location=lambda storage, loc: storage,
                )
            )
        self.optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage)
        )
