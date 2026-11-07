# Modified from EPyMARL for OA-QMIX and opponent-profile research.
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class OpponentAuxHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, style_classes, n_agents):
        super(OpponentAuxHead, self).__init__()
        self.style_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, style_classes),
        )
        self.focus_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents),
        )
        self.danger_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, opp_profile):
        x = th.cat([state, opp_profile], dim=-1)
        return {
            "style_logits": self.style_head(x),
            "focus_logits": self.focus_head(x),
            "danger_logits": self.danger_head(x),
        }


def masked_aux_loss(outputs, labels, masks):
    total = outputs["style_logits"].new_tensor(0.0)
    metrics = {"opponent_aux_loss": 0.0, "opponent_style_acc": 0.0, "focus_target_acc": 0.0}
    terms = 0

    style_mask = masks.get("style")
    if style_mask is not None and style_mask.any():
        style_labels = labels["style"][style_mask].long()
        style_logits = outputs["style_logits"][style_mask]
        style_loss = F.cross_entropy(style_logits, style_labels)
        total = total + style_loss
        metrics["opponent_style_acc"] = (
            style_logits.argmax(dim=-1).eq(style_labels).float().mean().item()
        )
        terms += 1

    focus_mask = masks.get("focus")
    if focus_mask is not None and focus_mask.any():
        focus_labels = labels["focus"][focus_mask].long()
        focus_logits = outputs["focus_logits"][focus_mask]
        focus_loss = F.cross_entropy(focus_logits, focus_labels)
        total = total + focus_loss
        metrics["focus_target_acc"] = (
            focus_logits.argmax(dim=-1).eq(focus_labels).float().mean().item()
        )
        terms += 1

    danger_mask = masks.get("danger")
    if danger_mask is not None and danger_mask.any():
        danger_labels = labels["danger"][danger_mask].float()
        danger_logits = outputs["danger_logits"][danger_mask].squeeze(-1)
        total = total + F.binary_cross_entropy_with_logits(danger_logits, danger_labels)
        terms += 1

    if terms:
        total = total / terms
        metrics["opponent_aux_loss"] = total.item()
    return total, metrics
