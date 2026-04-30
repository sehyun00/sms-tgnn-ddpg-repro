from main import load_config, run_backtest_mode, configure_reproducibility


def main() -> None:
    """Compatibility wrapper for `python main.py --mode backtest`."""
    config = load_config("config/config.yaml")
    configure_reproducibility(config)
    run_backtest_mode(config)


if __name__ == "__main__":
    main()
