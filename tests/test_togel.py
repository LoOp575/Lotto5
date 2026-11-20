"""Tes pipeline data Togel.py (tanpa TensorFlow)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import Togel


def tulis_csv(tmp_path: Path, baris: list[str]) -> Path:
    path = tmp_path / "data.csv"
    path.write_text("\n".join(baris) + "\n", encoding="utf-8")
    return path


VALID = [
    "4959\t9219\t9689\t6086\t4909\t4106\t5154",
    "3286\t8242\t7079\t0745\t8021\t6172\t5695",
    "2413\t6681\t4283\t3249\t4156\t3313\t9859",
    "0425\t3023\t5380\t2814\t1563\t3987\t7116",
]


# --------------------------------------------------------------------------- #
# pemuatan data
# --------------------------------------------------------------------------- #
def test_baca_tsv_dan_pertahankan_nol_di_depan(tmp_path):
    df = Togel.muat_data(tulis_csv(tmp_path, VALID))

    assert df.shape == (4, 7)
    assert df.iloc[1, 3] == "0745"          # bukan 745
    assert df.iloc[3, 0] == "0425"


def test_baris_tidak_lengkap_dibuang_dengan_peringatan(tmp_path, capsys):
    baris = VALID + ["7301\t8951\t4105\t5817"]      # baris 5 cuma 4 kolom
    df = Togel.muat_data(tulis_csv(tmp_path, baris))

    assert len(df) == 4
    assert "baris 5" in capsys.readouterr().err


def test_sel_bukan_empat_digit_ditolak(tmp_path):
    baris = VALID + ["73018\t8951\t4105\t5817\t0001\t0002\t0003"]
    with pytest.raises(ValueError, match="bukan 4 digit"):
        Togel.muat_data(tulis_csv(tmp_path, baris))


def test_file_hilang_pesan_jelas(tmp_path):
    with pytest.raises(FileNotFoundError, match="data tidak ditemukan"):
        Togel.muat_data(tmp_path / "tidak_ada.csv")


def test_file_kosong_ditolak(tmp_path):
    with pytest.raises(ValueError, match="tidak berisi"):
        Togel.muat_data(tulis_csv(tmp_path, ["7301\t8951"]))


# --------------------------------------------------------------------------- #
# fitur
# --------------------------------------------------------------------------- #
def test_ke_digit_bentuk_dan_isi(tmp_path):
    df = Togel.muat_data(tulis_csv(tmp_path, VALID))
    d = Togel.ke_digit(df)

    assert d.shape == (4, 7, 4)
    assert d[1, 3].tolist() == [0, 7, 4, 5]         # nol di depan ikut
    assert "".join(map(str, d[0].ravel())) == "".join(VALID[0].split("\t"))


def test_one_hot_digit_280_biner(tmp_path):
    df = Togel.muat_data(tulis_csv(tmp_path, VALID))
    vektor = Togel.one_hot_digit(Togel.ke_digit(df)[0])

    assert vektor.shape == (Togel.N_FITUR,) == (280,)
    assert vektor.sum() == 28                       # 7 posisi x 4 digit
    assert set(np.unique(vektor)) == {0.0, 1.0}
    # digit pertama 4 -> indeks 0*10 + 4
    assert vektor[4] == 1.0
    assert vektor[0] == 0.0


def test_buat_dataset_pasangan_lag_satu(tmp_path):
    df = Togel.muat_data(tulis_csv(tmp_path, VALID))
    X, y = Togel.buat_dataset(df, posisi=1)

    assert X.shape == (3, 280)                      # satu kurang dari jumlah baris
    assert y.tolist() == [3286, 2413, 425]          # posisi 1 baris 2,3,4
    # fitur baris t = undian t-1
    assert X[0].sum() == 28


def test_butuh_minimal_dua_undian(tmp_path):
    df = Togel.muat_data(tulis_csv(tmp_path, VALID[:1]))
    with pytest.raises(ValueError, match="minimal 2 undian"):
        Togel.buat_dataset(df, posisi=1)


def test_posisi_di_luar_rentang_ditolak(tmp_path):
    df = Togel.muat_data(tulis_csv(tmp_path, VALID))
    for posisi in (0, 8):
        with pytest.raises(ValueError, match="posisi"):
            Togel.buat_dataset(df, posisi=posisi)


# --------------------------------------------------------------------------- #
# split & metrik
# --------------------------------------------------------------------------- #
def test_split_kronologis():
    X = np.arange(40).reshape(20, 2)
    y = np.arange(20)

    X_tr, X_val, y_tr, y_val = Togel.bagi_data(X, y, test_size=0.25)

    assert len(X_tr) == 15 and len(X_val) == 5
    assert y_tr.max() < y_val.min()                 # data lama di latih
    assert X_val[0, 0] == 30


def test_split_menolak_data_terlalu_sedikit():
    with pytest.raises(ValueError, match="terlalu sedikit"):
        Togel.bagi_data(np.zeros((1, 3)), np.zeros(1), 0.5)


def test_top_k_akurasi():
    y = np.array([3, 7])
    skor = np.zeros((2, 10))
    skor[0, 3] = 1.0
    skor[1, 7] = 0.5

    assert Togel.top_k_akurasi(y, skor, 1) == 1.0
    assert Togel.top_k_akurasi(y, skor, 5) == 1.0

    skor[0, 3] = 0.0                                # kelas benar tidak teratas
    assert Togel.top_k_akurasi(y, skor, 1) == 0.5


def test_top_k_dengan_k_lebih_besar_dari_kelas():
    y = np.array([2])
    skor = np.zeros((1, 3))

    assert Togel.top_k_akurasi(y, skor, 10) == 1.0   # k di-clip ke jumlah kelas


def test_baseline_mayoritas():
    y_train = np.array([5, 5, 5, 9, 1])
    y_val = np.array([5, 2])

    kelas, akurasi = Togel.baseline_mayoritas(y_train, y_val)
    assert kelas == 5
    assert akurasi == 0.5


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_peringatan_kelas_tidak_beririsan(tmp_path, capsys):
    # 4 undian unik -> tidak ada kelas yang sama antara latih dan validasi
    path = tulis_csv(tmp_path, VALID)

    assert Togel.main(["--data", str(path), "--dry-run"]) == 0
    err = capsys.readouterr().err
    assert "tidak ada satu pun kelas" in err


def test_dry_run_tidak_butuh_tensorflow(tmp_path, capsys):
    path = tulis_csv(tmp_path, VALID)

    assert Togel.main(["--data", str(path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "fitur          : 280 biner" in out
    assert "TensorFlow tidak diimpor" in out
