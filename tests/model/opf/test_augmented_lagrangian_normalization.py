import warnings
import math

import pytest
import torch

from lumina.model.opf.augmented_lagrangian import AugmentedLagrangianACOPF


@pytest.fixture
def network_inputs():
    y_idx = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    y_val = torch.zeros(2, dtype=torch.float32)
    y_sparse = torch.sparse_coo_tensor(y_idx, y_val, (2, 2)).coalesce()

    return {
        'Y_real_sparse': y_sparse,
        'Y_imag_sparse': y_sparse,
        'line_edge_index': torch.tensor([[0], [1]], dtype=torch.long),
        'line_y_ff_real': torch.tensor([1.0]),
        'line_y_ff_imag': torch.tensor([0.0]),
        'line_y_ft_real': torch.tensor([0.0]),
        'line_y_ft_imag': torch.tensor([0.0]),
        'line_y_tf_real': torch.tensor([0.0]),
        'line_y_tf_imag': torch.tensor([0.0]),
        'line_y_tt_real': torch.tensor([0.0]),
        'line_y_tt_imag': torch.tensor([0.0]),
        'line_limits': torch.tensor([0.5]),
    }


def _build_data(include_power: bool, include_line: bool, network_inputs: dict):
    data = {}
    if include_power:
        data.update(
            {
                'pd': torch.tensor([1.0]),
                'qd': torch.tensor([3.0]),
                'gen_bus_indices': torch.tensor([0], dtype=torch.long),
                'load_bus_indices': torch.tensor([1], dtype=torch.long),
            }
        )
    if include_line:
        data['line_edge_index'] = network_inputs['line_edge_index']
    return data


def _build_predictions():
    return {
        'bus': torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float32),
        'generator': torch.tensor([[2.0, 4.0]], dtype=torch.float32),
    }


def _expected_power_raw():
    return torch.tensor([2.0, -1.0, 4.0, -3.0], dtype=torch.float32)


def _expected_line_raw():
    return torch.tensor([math.sqrt(0.75)], dtype=torch.float32)




def _setup_lagrangian(normalize_by_rms: bool, normalize_by_size: bool, network_inputs: dict, include_line: bool, include_power: bool):
    lag = AugmentedLagrangianACOPF(normalize_by_rms=normalize_by_rms, normalize_by_size=normalize_by_size)
    kwargs = dict(network_inputs)
    if not include_power:
        kwargs.update({
            'Y_real_sparse': None,
            'Y_imag_sparse': None,
        })
    if not include_line:
        kwargs.update(
            {
                'line_edge_index': None,
                'line_y_ff_real': None,
                'line_y_ff_imag': None,
                'line_y_ft_real': None,
                'line_y_ft_imag': None,
                'line_y_tf_real': None,
                'line_y_tf_imag': None,
                'line_y_tt_real': None,
                'line_y_tt_imag': None,
                'line_limits': None,
            }
        )
    lag.set_network_parameters(**kwargs)
    return lag

def _normalize(vec: torch.Tensor, by_rms: bool, by_size: bool) -> torch.Tensor:
    out = vec.clone()
    if by_rms:
        out = out / (torch.sqrt(torch.mean(out**2)) + 1e-8)
    if by_size:
        out = out / math.sqrt(out.numel())
    return out


@pytest.mark.parametrize('normalize_by_rms,normalize_by_size', [(False, True), (True, True), (False, False), (True, False)])
@pytest.mark.parametrize('include_power,include_line', [(True, False), (False, True), (True, True)])
def test_constraint_normalization_combinations(network_inputs, normalize_by_rms, normalize_by_size, include_power, include_line):
    lag = _setup_lagrangian(
        normalize_by_rms,
        normalize_by_size,
        network_inputs,
        include_line=include_line,
        include_power=include_power,
    )

    predictions = _build_predictions()
    data = _build_data(include_power=include_power, include_line=include_line, network_inputs=network_inputs)

    eq_constraints, line_constraints = lag.compute_constraint_components(predictions, data)

    expected_parts = []
    if include_power:
        expected_parts.append(_normalize(_expected_power_raw(), normalize_by_rms, normalize_by_size))
    if include_line:
        expected_parts.append(_normalize(_expected_line_raw(), normalize_by_rms, normalize_by_size))

    if include_power:
        assert torch.allclose(eq_constraints, expected_parts[0], atol=1e-6)
    else:
        assert eq_constraints.numel() == 0

    if include_line:
        idx = 1 if include_power else 0
        assert torch.allclose(line_constraints, expected_parts[idx], atol=1e-6)
    else:
        assert line_constraints.numel() == 0

    combined = lag.compute_constraints(predictions, data)
    if expected_parts:
        expected_combined = torch.cat(expected_parts)
        assert torch.allclose(combined, expected_combined, atol=1e-6)

    if include_line:
        assert lag._last_line_limit_rms is not None
        assert lag._last_line_limit_normalized_rms is not None


def test_legacy_normalize_constraints_true_maps_predictably():
    with pytest.warns(DeprecationWarning, match='normalize_constraints'):
        lag = AugmentedLagrangianACOPF(normalize_constraints=True)

    assert lag.normalize_by_rms is True
    assert lag.normalize_by_size is True


def test_legacy_normalize_constraints_false_maps_predictably():
    with pytest.warns(DeprecationWarning, match='normalize_constraints'):
        lag = AugmentedLagrangianACOPF(normalize_constraints=False)

    assert lag.normalize_by_rms is False
    assert lag.normalize_by_size is False


def test_new_keys_take_precedence_over_legacy_key():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        lag = AugmentedLagrangianACOPF(
            normalize_by_rms=True,
            normalize_by_size=False,
            normalize_constraints=False,
        )

    assert not any(issubclass(w.category, DeprecationWarning) for w in record)
    assert lag.normalize_by_rms is True
    assert lag.normalize_by_size is False
