"""Unit tests for configs/training_config.py's new SP-1 fields."""

from configs.training_config import DataConfig, get_training_config


def test_get_training_config_reads_data_section():
    cfg = get_training_config("configs/config.yaml")
    assert cfg.data.target_corpus_size == 1_000_000
    assert cfg.data.hard_neg_max_queries == 150_000
    assert cfg.data.hard_negatives_top_k == 100
    assert cfg.data.max_train_queries == 400_000
    assert cfg.data.max_dev_queries == 6980


def test_get_training_config_reads_updated_two_tower_section():
    cfg = get_training_config("configs/config.yaml")
    # Values reflect the committed full-run config (clean venv + AMP on an 8GB
    # RTX 4060): batch 16 (batch 32 AMP thrashed VRAM), 3 epochs, per-epoch
    # checkpoint-eval index capped at 30k distractors. See configs/config.yaml.
    assert cfg.two_tower.hard_negatives_per_query == 5
    assert cfg.two_tower.batch_size == 16
    assert cfg.two_tower.epochs == 3
    assert cfg.two_tower.eval_max_distractors == 30_000


def test_data_config_defaults_without_yaml():
    cfg = DataConfig()
    assert cfg.target_corpus_size == 1_000_000
    assert cfg.hard_neg_max_queries == 150_000
