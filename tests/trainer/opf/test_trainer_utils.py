"""Tests for lumina.trainer.opf.utils pure helpers (no DDP/IO required)."""

import pytest

from lumina.trainer.opf.utils import (
    _is_main_process,
    apply_nested,
    build_hetero_model_spec,
    get_case_name_mapping,
    parse_case_name,
    parse_cases_arg,
    resolve_hetero_model_type,
    select_cuda_device_index,
)


# -- _is_main_process -----------------------------------------------------------

def test_is_main_process_returns_true_outside_ddp():
    # No process group has been initialized in the test process.
    assert _is_main_process() is True


# -- get_case_name_mapping ------------------------------------------------------

def test_get_case_name_mapping_returns_copy():
    m1 = get_case_name_mapping()
    m2 = get_case_name_mapping()
    assert m1 == m2
    m1['case14'] = 'mutated'
    assert get_case_name_mapping()['case14'] == 'pglib_opf_case14_ieee'


# -- parse_case_name ------------------------------------------------------------

@pytest.mark.parametrize("short,full", [
    ('case14', 'pglib_opf_case14_ieee'),
    ('case30', 'pglib_opf_case30_ieee'),
    ('case2000', 'pglib_opf_case2000_goc'),
    ('case13659', 'pglib_opf_case13659_pegase'),
])
def test_parse_case_name_short_to_full(short, full):
    assert parse_case_name(short) == full


def test_parse_case_name_passes_through_qualified_names():
    assert parse_case_name('pglib_opf_case14_ieee') == 'pglib_opf_case14_ieee'


def test_parse_case_name_accepts_numeric_only():
    assert parse_case_name('14') == 'pglib_opf_case14_ieee'


def test_parse_case_name_unknown_raises():
    with pytest.raises(ValueError, match='Invalid case name'):
        parse_case_name('case99999')


# -- parse_cases_arg ------------------------------------------------------------

def test_parse_cases_arg_plain_list():
    assert parse_cases_arg(['case14', 'case30']) == ['case14', 'case30']


def test_parse_cases_arg_comma_separated():
    assert parse_cases_arg(['case14,case30,case57']) == ['case14', 'case30', 'case57']


def test_parse_cases_arg_json_list():
    assert parse_cases_arg(['["case14", "case30"]']) == ['case14', 'case30']


def test_parse_cases_arg_strips_whitespace_and_skips_empty():
    assert parse_cases_arg(['  case14  ', '', '  ']) == ['case14']


def test_parse_cases_arg_mixed_forms():
    assert parse_cases_arg(['case14', 'case30,case57']) == ['case14', 'case30', 'case57']


# -- resolve_hetero_model_type --------------------------------------------------

@pytest.mark.parametrize("alias,canonical", [
    ('HeteroGNN', 'HeteroGNN'),
    ('heterognn', 'HeteroGNN'),
    ('OPFHeteroGNN', 'HeteroGNN'),
    ('rgat', 'RGAT'),
    ('HEAT', 'HEAT'),
    ('  hgt  ', 'HGT'),
])
def test_resolve_hetero_model_type_case_insensitive(alias, canonical):
    assert resolve_hetero_model_type(model_type=alias) == canonical


def test_resolve_hetero_model_type_from_class_path():
    path = 'lumina.model.opf.hetero_model.OPFHeteroGNN'
    assert resolve_hetero_model_type(model_class_path=path) == 'HeteroGNN'


def test_resolve_hetero_model_type_default_when_empty():
    assert resolve_hetero_model_type(model_type=None) == 'HeteroGNN'
    assert resolve_hetero_model_type(model_type='') == 'HeteroGNN'


def test_resolve_hetero_model_type_unknown_raises():
    with pytest.raises(ValueError, match='Unsupported hetero model type'):
        resolve_hetero_model_type(model_type='not_a_model')


def test_resolve_hetero_model_type_unknown_class_path_raises():
    with pytest.raises(ValueError, match='Unsupported hetero model class path'):
        resolve_hetero_model_type(model_class_path='pkg.mod.MysteryModel')


# -- select_cuda_device_index ---------------------------------------------------

def test_select_cuda_device_index_single_visible_device_uses_zero():
    assert select_cuda_device_index(local_rank=3, visible_device_count=1) == 0


def test_select_cuda_device_index_multi_visible_uses_local_rank():
    assert select_cuda_device_index(local_rank=2, visible_device_count=4) == 2


def test_select_cuda_device_index_negative_rank_clamps_to_zero():
    assert select_cuda_device_index(local_rank=-1, visible_device_count=4) == 0


def test_select_cuda_device_index_invalid_rank_defaults_zero():
    assert select_cuda_device_index(local_rank="bad", visible_device_count=4) == 0


def test_select_cuda_device_index_out_of_range_raises():
    with pytest.raises(ValueError, match="LOCAL_RANK=4 exceeds visible CUDA device count"):
        select_cuda_device_index(local_rank=4, visible_device_count=4)


# -- apply_nested ---------------------------------------------------------------

def test_apply_nested_sets_top_level_key():
    d = {}
    apply_nested(d, 'lr', 1e-3)
    assert d == {'lr': 1e-3}


def test_apply_nested_creates_intermediate_dicts():
    d = {}
    apply_nested(d, 'optimizer.AdamW.lr', 1e-3)
    assert d == {'optimizer': {'AdamW': {'lr': 1e-3}}}


def test_apply_nested_overrides_existing_value():
    d = {'training': {'max_epochs': 10}}
    apply_nested(d, 'training.max_epochs', 50)
    assert d['training']['max_epochs'] == 50


def test_apply_nested_preserves_siblings():
    d = {'training': {'max_epochs': 10}}
    apply_nested(d, 'training.lr', 1e-4)
    assert d == {'training': {'max_epochs': 10, 'lr': 1e-4}}


# -- build_hetero_model_spec ----------------------------------------------------

def _metadata():
    node_types = ['bus', 'generator', 'load', 'shunt']
    edge_types = [('bus', 'ac_line', 'bus'), ('generator', 'generator_link', 'bus')]
    return node_types, edge_types


def _input_channels():
    return {'bus': 4, 'generator': 11, 'load': 2, 'shunt': 2}


def test_build_hetero_model_spec_for_heterognn_uses_config():
    cls, kwargs, _config, used_fallback = build_hetero_model_spec(
        model_type='HeteroGNN',
        metadata=_metadata(),
        input_channels=_input_channels(),
        models_config={'HeteroGNN': {'hidden_channels': 128, 'backend': 'gat', 'num_layers': 4}},
    )
    from lumina.model.opf.hetero_model import OPFHeteroGNN
    assert cls is OPFHeteroGNN
    assert kwargs['hidden_channels'] == 128
    assert kwargs['backend'] == 'gat'
    assert kwargs['num_layers'] == 4
    assert kwargs['out_channels'] == 2
    assert kwargs['metadata'] == _metadata()
    assert kwargs['input_channels'] == _input_channels()
    assert used_fallback is False


def test_build_hetero_model_spec_falls_back_to_heterognn():
    cls, kwargs, _config, used_fallback = build_hetero_model_spec(
        model_type='HGT',
        metadata=_metadata(),
        input_channels=_input_channels(),
        models_config={'HeteroGNN': {'hidden_channels': 64, 'num_layers': 2}},
    )
    from lumina.model.opf.hetero_model import HGT
    assert cls is HGT
    assert kwargs['hidden_channels'] == 64
    assert used_fallback is True
    # HGT-specific default for num_heads should be applied
    assert kwargs['num_heads'] == 1


def test_build_hetero_model_spec_applies_default_kwargs_when_no_config():
    cls, kwargs, config, used_fallback = build_hetero_model_spec(
        model_type='HEAT',
        metadata=_metadata(),
        input_channels=_input_channels(),
        models_config={},
    )
    from lumina.model.opf.hetero_model import HEAT
    assert cls is HEAT
    assert kwargs['hidden_channels'] == 64
    assert kwargs['num_layers'] == 3
    assert kwargs['backend'] == 'sage'
    assert kwargs['attention_heads'] == 1
    assert config == {}
    assert used_fallback is False


def test_build_hetero_model_spec_unknown_type_raises():
    with pytest.raises(ValueError):
        build_hetero_model_spec(
            model_type='not_a_model',
            metadata=_metadata(),
            input_channels=_input_channels(),
            models_config={},
        )
