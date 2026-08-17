from torch import nn

from .heads import BasicActorHead, BaseCriticHead 


class ActorCritic(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        actor_head: BasicActorHead,
        critic_head: BaseCriticHead,
    ) -> None:
        super().__init__()
        self.base = base
        self.actor_head = actor_head
        self.critic_head = critic_head
        
    def forward(
        self,
        obs,
        action_info,
        omit_value = False,
    ) -> ActorCriticOut:
        ...