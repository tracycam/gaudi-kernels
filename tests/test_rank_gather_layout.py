"""Reject layouts that lose, repeat or reorder ranks before creating groups."""
import pytest
from gaudi_kernels.serving.distributed.rank_gather import GatherLayout


def test_route_can_exchange_different_peers_without_reordering_rank_bytes():
    layout=GatherLayout(((0,1,2,3),(4,5,6,7)),((0,5),(1,6),(2,7),(3,4)))
    layout.validate(tuple(range(8)))
    for exchange in layout.exchange_groups:
        result=tuple(rank for representative in exchange
                     for local in layout.local_groups if representative in local for rank in local)
        assert result==tuple(range(8))


def test_non_eight_rank_layout():
    GatherLayout(((1,3),(5,7),(9,11)),((1,7,9),(3,5,11))).validate((1,3,5,7,9,11))


@pytest.mark.parametrize('layout',[
    GatherLayout(((0,1),(2,3)),((0,2),(0,3))),
    GatherLayout(((0,2),(1,3)),((0,1),(2,3))),
    GatherLayout(((0,1),(2,3)),((0,1),(2,3))),
    GatherLayout(((0,1),(2,3)),((2,0),(1,3))),
    GatherLayout(((0,1,2),(3,)),((0,3),(1,2))),
])
def test_invalid_routes_are_rejected(layout):
    with pytest.raises(ValueError):
        layout.validate((0,1,2,3))


def test_deployment_plan_round_trip_preserves_arithmetic_contract():
    from gaudi_kernels.engine import EngineConfig
    base=EngineConfig()
    config=EngineConfig.from_dict({'runtime':{'communication':{
        'gather':'two_stage','local_groups':[[0,1,2,3],[4,5,6,7]],
        'exchange_groups':[[0,5],[1,6],[2,7],[3,4]]}}})
    assert EngineConfig.from_dict(config.to_dict())==config
    assert config.arithmetic_fingerprint==base.arithmetic_fingerprint
    assert base.runtime.communication.gather=='direct'


def test_invalid_deployment_plan_fails_before_device_acquisition():
    from gaudi_kernels.engine import EngineConfig,ConfigError
    with pytest.raises(ConfigError):
        EngineConfig.from_dict({'runtime':{'communication':{
            'gather':'two_stage','local_groups':[[0,1,2,3],[4,5,6,7]],
            'exchange_groups':[[0,4],[0,5],[2,6],[3,7]]}}})
    with pytest.raises(ConfigError):
        EngineConfig.from_dict({'runtime':{'communication':{'local_groups':[[0,1],[2,3]]}}})
