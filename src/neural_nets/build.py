import math
from typing import Literal, Annotated, Field
from typing_extensions import assert_never

from torch import nn
from pydantic import BaseModel, field_validator

from .types import ActivationFactory
from .actor_critic import ActorCriticAttnBase, ActorCriticConvBase, ActorCritic
from .heads import BasicActorHead, ZeroSumCriticHead


class ActorCriticAttnConfig(BaseModel):
    model_arch: Literal["attn"]
    d_model: int
    num_heads: int
    n_blocks: int
    dropout: Annotated[float | None, Field(gt=0.0, le=0.2)] = None
    
    @property
    def d_mlp(self) -> int:
        return self.d_model * 4
    
    @field_validator("d_model")
    @classmethod
    def _validate_d_model(cls, d_model: int) -> int:
        if not math.log2(d_model).is_integer():
            raise ValueError("d_model should be power of 2")
        return d_model
    
class ActorCriticConvConfig(BaseModel):
    model_arch: Literal["conv"]
    d_model: int
    n_blocks: int
    kernel_size: int = 3
    dropout: Annotated[float | None, Field(gt=0.0, le=0.2)] = None
    
    @field_validator("d_model")
    @classmethod
    def _validate_d_model(cls, d_model: int) -> int:
        if not math.log2(d_model).is_integer():
            raise ValueError("d_model should be power of 2")
        return d_model
    
ActorCriticConfigT = Annotated[
    ActorCriticAttnConfig | ActorCriticConvConfig,
    Field(discriminator="model_arch"),
]


class ActorCriticConfigWrapper(BaseModel):
    config: ActorCriticConfigT

def build_base(
    spatial_in_channels: int,
    global_in_channels: int,
    config: ActorCriticConfigT,
    activation: ActivationFactory,
) -> nn.Module:
    if isinstance(config, ActorCriticAttnConfig):
        return ActorCriticAttnBase(
            spatial_in_channels=spatial_in_channels,
            global_in_channels=global_in_channels,
            d_model=config.d_model,
            d_mlp=config.d_mlp,
            num_heads=config.num_heads,
            n_blocks=config.n_blocks,
            activation=activation,
            dropout=config.dropout,
        )
    
    if isinstance(config, ActorCriticConvConfig):
        ActorCriticConvBase(
            spatial_in_channels=spatial_in_channels,
            global_in_channels=global_in_channels,
            d_model=config.d_model,
            n_blocks=config.n_blocks,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
            activation=activation,
        )
        
    assert_never(config)
  
def build_actor_critic(
  spatial_in_channels: int,
  global_in_channels: int,
  n_main_actions: int,
  reward_space,
  config,
):
    activation = ActivationFactory
    
    base = build_base(
        spatial_in_channels=spatial_in_channels,
        global_in_channels=global_in_channels,
        config=config,
        activation=activation,
    )
    
    actor_head = BasicActorHead(
        d_model=config.d_model,
        activation=activation,
        n_main_actions=n_main_actions,
    )
    
    critic_head = ZeroSumCriticHead(
        d_model=config.d_model,
        activation=activation,
    )
    
    model: nn.Module = ActorCritic(
        base=base,
        actor_head=actor_head,
        critic_head=critic_head,
    )