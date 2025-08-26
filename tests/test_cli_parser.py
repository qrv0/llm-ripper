from llm_ripper.cli import create_parser


def test_cli_has_subcommands_and_flags():
    p = create_parser()
    # Only test parsing structure; do not execute handlers
    ns = p.parse_args(["extract", "--model", "m", "--output-dir", "o"])
    assert ns.command == "extract"
    ns = p.parse_args(["capture", "--model", "m", "--output-file", "a.h5"])
    assert ns.command == "capture"
    ns = p.parse_args(["analyze", "--knowledge-bank", "kb", "--output-dir", "out"])
    assert ns.command == "analyze"
    ns = p.parse_args(
        ["transplant", "--source", "kb", "--target", "m", "--output-dir", "out"]
    )
    assert ns.command == "transplant"
    ns = p.parse_args(["validate", "--model", "m", "--output-dir", "out"])
    assert ns.command == "validate"
    ns = p.parse_args(["inspect", "--knowledge-bank", "kb"])
    assert ns.command == "inspect"
