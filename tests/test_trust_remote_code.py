import sys
import pytest

from llm_ripper.cli import create_parser


def test_trust_remote_code_requires_yes(monkeypatch, capsys):
    parser = create_parser()
    # Simulate: top-level command with a subcommand, here use 'extract' minimal args
    args = parser.parse_args(
        ["extract", "--model", "dummy", "--output-dir", "./out", "--trust-remote-code"]
    )
    # The CLI applies overrides inside command; here we directly test the parser behavior through apply_cli_overrides
    # Because apply_cli_overrides is internal, we simulate the failure by calling the CLI path that checks the flag
    # This test asserts that the check prints an error and exits non-zero
    from llm_ripper.cli import apply_cli_overrides, ConfigManager

    cfg = ConfigManager(None)
    with pytest.raises(SystemExit) as ex:
        apply_cli_overrides(cfg, args)
    assert ex.value.code == 1
    captured = capsys.readouterr()
    assert "requires confirmation" in captured.out
