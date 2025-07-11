import pandas as pd
from pathlib import Path

from titanic_influence.influence_first_order import compute_influence as compute_fo
from titanic_influence.influence_tracin import compute_influence as compute_tr


def test_influence_pipeline(tmp_path: Path) -> None:
    fo_dir = tmp_path / "fo"
    tr_dir = tmp_path / "tr"

    fo_res = compute_fo([0], k=5, output_dir=fo_dir)
    tr_res = compute_tr([0], k=5, output_dir=tr_dir)

    fo_csv = fo_dir / "test_0.csv"
    tr_csv = tr_dir / "test_0.csv"
    assert fo_csv.exists() and tr_csv.exists()

    fo_df = pd.read_csv(fo_csv)
    tr_df = pd.read_csv(tr_csv)

    assert len(fo_df) >= 5 and len(tr_df) >= 5
    assert list(fo_df.columns) == ["train_index", "score", "rank"]
    assert list(tr_df.columns) == ["train_index", "score", "rank"]

    # Ensure mapping return values contain DataFrames
    assert 0 in fo_res and 0 in tr_res
    assert isinstance(fo_res[0], pd.DataFrame)
    assert isinstance(tr_res[0], pd.DataFrame)
