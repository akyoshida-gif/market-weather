from mw.ingest.us import cboe


def test_spx_history_single_value_file_parses_as_close() -> None:
    df = cboe.parse("SPX", b"DATE,SPX\n01/02/1975,70.230000\n01/03/1975,70.710000\n")
    assert df.height == 2 and df["close"].to_list() == [70.23, 70.71]
    assert df["open"].null_count() == 2
