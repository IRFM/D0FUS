"""Blanket layer allocation (``radial_layers_effective``, M. Fletcher 2026):
closure of the inboard and outboard stacks, the priority order
mandatory layers > shields > breeder and the compressed stack of thin builds.
Structural checks only, no frozen values.
"""
import numpy as np
import pytest

from D0FUS_BIB.D0FUS_parameterization import (
    radial_build_feasibility,
    radial_layers_effective,
)

CONCEPTS = ('HCPB', 'HCLL', 'DCLL', 'SCLL', 'SCLV', 'F-LIB')


def _width(layers, side, roles):
    return sum(L['delta_' + side] for L in layers if L['role'] in roles)


@pytest.mark.parametrize("concept", CONCEPTS)
@pytest.mark.parametrize("b, Delta_TF", [(1.2, 0.0), (1.2, 1.5), (0.6, 0.4), (2.0, 0.8)])
def test_stacks_close_on_both_sides(concept, b, Delta_TF):
    layers = radial_layers_effective(concept, b, Delta_TF)
    assert layers, 'feasible build expected'
    np.testing.assert_allclose(_width(layers, 'ib', {L['role'] for L in layers}), b, atol=1e-9)
    np.testing.assert_allclose(_width(layers, 'ob', {L['role'] for L in layers}), b + Delta_TF,
                               atol=1e-9)


def test_inboard_and_outboard_shield_targets():
    layers = radial_layers_effective('HCPB', 1.2, 1.0, delta_BB_IB=0.3, delta_BB_OB=0.3,
                                     delta_shield_IB=0.2, delta_shield_OB=0.5)
    for side, target in (('ib', 0.2), ('ob', 0.5)):
        np.testing.assert_allclose(_width(layers, side, {'shield_HT', 'shield_LT'}), target)
        np.testing.assert_allclose(_width(layers, side, {'shield_HT'}),
                                   _width(layers, side, {'shield_LT'}))


def test_shields_before_breeder_and_structure():
    # Room for the mandatory layers and part of the shield only: no breeder,
    # no structure, shield limited by the space left
    b = radial_build_feasibility('HCPB', 0.0)['mandatory_ib'] + 0.1
    layers = radial_layers_effective('HCPB', b, 0.0)
    np.testing.assert_allclose(_width(layers, 'ib', {'shield_HT', 'shield_LT'}), 0.1)
    assert _width(layers, 'ib', {'breeder', 'structure', 'multiplier'}) == 0.0


def test_compressed_stack_below_mandatory_layers():
    # As in D0FUS 3.2: mandatory layers scaled to fit, no shield, no breeder
    b = radial_build_feasibility('HCPB', 0.0)['mandatory_ib'] - 0.05
    assert radial_build_feasibility('HCPB', b)['feasible']
    layers = radial_layers_effective('HCPB', b, 0.0)
    np.testing.assert_allclose(_width(layers, 'ib', {L['role'] for L in layers}), b, atol=1e-9)
    assert _width(layers, 'ib', {'shield_HT', 'shield_LT', 'breeder', 'structure'}) == 0.0
    assert 0.0 < layers[0]['stack_scale'] < 1.0


def test_no_radial_space_has_no_layers():
    assert not radial_build_feasibility('HCPB', 0.0)['feasible']
    assert radial_layers_effective('HCPB', 0.0, 0.0) == []
