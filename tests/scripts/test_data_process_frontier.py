import os

from scripts.data_process_frontier import _parse_int_list, _parse_str_list, get_rank_size


def test_parse_int_list_default_on_empty():
    default = [0, 1, 2]
    assert _parse_int_list(None, default) == default
    assert _parse_int_list("", default) == default


def test_parse_int_list_accepts_csv_and_space_separated():
    assert _parse_int_list("1,2,3", []) == [1, 2, 3]
    assert _parse_int_list("4 5 6", []) == [4, 5, 6]
    assert _parse_int_list("7, 8 9", []) == [7, 8, 9]


def test_parse_str_list_default_on_empty():
    default = ["a", "b"]
    assert _parse_str_list(None, default) == default
    assert _parse_str_list("", default) == default


def test_parse_str_list_accepts_csv_and_space_separated():
    assert _parse_str_list("case14,case118", []) == ["case14", "case118"]
    assert _parse_str_list("case14 case118", []) == ["case14", "case118"]
    assert _parse_str_list("case14, case118 case300", []) == ["case14", "case118", "case300"]


def test_get_rank_size_prefers_slurm_env(monkeypatch):
    monkeypatch.setenv("SLURM_PROCID", "3")
    monkeypatch.setenv("SLURM_NTASKS", "16")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "8")

    rank, size = get_rank_size()

    assert rank == 3
    assert size == 16


def test_get_rank_size_falls_back_to_rank_world_size(monkeypatch):
    monkeypatch.delenv("SLURM_PROCID", raising=False)
    monkeypatch.delenv("SLURM_NTASKS", raising=False)
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "4")

    rank, size = get_rank_size()

    assert rank == 2
    assert size == 4


def test_get_rank_size_defaults_to_single_process(monkeypatch):
    for key in ("SLURM_PROCID", "SLURM_NTASKS", "RANK", "WORLD_SIZE"):
        monkeypatch.delenv(key, raising=False)

    rank, size = get_rank_size()

    assert rank == 0
    assert size == 1
