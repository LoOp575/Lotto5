"""Tes untuk bench/header_bench.py (tanpa jaringan sungguhan)."""

from __future__ import annotations

from collections import Counter

import pytest

import header_bench as hb


# --------------------------------------------------------------------------- #
# pembuat header
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [1, 2, 100, 999, 1000])
def test_normal_jumlah_unik_dan_valid(n):
    hdrs = hb.headers_normal(n)

    assert len(hdrs) == n
    assert len({k for k, _ in hdrs}) == n            # tidak ada duplikat
    for nama, _ in hdrs:
        assert nama.isascii()
        assert "_" not in nama                       # underscore = invalid di nginx
        assert 0x20 < ord(nama[0]) < 0x7F


@pytest.mark.parametrize("n", [1, 2, 3, 100, 300, 1000])
def test_collision_jumlah_unik_dan_karakter(n):
    hdrs = hb.headers_collision(n)

    assert len(hdrs) == n
    assert len({k for k, _ in hdrs}) == n
    assert sum(1 for k, _ in hdrs if set(k) <= {"_", "^", "~"}) == n


def test_collision_selalu_menghasilkan_n_nama_untuk_n_bukan_pangkat_dua():
    # k = bit_length(n-1) harus cukup untuk 2**k >= n
    for n in (3, 5, 17, 129, 1001):
        assert len({k for k, _ in hb.headers_collision(n)}) == n


def test_lebar_seragam_menyamakan_byte_kedua_varian():
    for n in (100, 300, 1000):
        lebar = hb.lebar_seragam([n])
        normal = hb.headers_normal(n, lebar)
        koll = hb.headers_collision(n, lebar)

        assert len({len(k) for k, _ in normal}) == 1
        assert len({len(k) for k, _ in koll}) == 1
        assert hb.header_bytes(normal) == hb.header_bytes(koll)


def test_lebar_seragam_menghindari_duplikat():
    # tanpa padding lebar, nama normal dan collision bisa bertabrakan panjang
    lebar = hb.lebar_seragam([10])
    assert len({k for k, _ in hb.headers_collision(10, lebar)}) == 10


def test_n_kurang_dari_satu_ditolak():
    for fn in (hb.headers_normal, hb.headers_collision):
        with pytest.raises(ValueError, match="n harus >= 1"):
            fn(0)


def test_header_bytes_menghitung_crlf():
    assert hb.header_bytes([("a", "1")]) == 1 + 1 + 4


# --------------------------------------------------------------------------- #
# statistik
# --------------------------------------------------------------------------- #
def test_percentile_nearest_rank():
    data = list(range(1, 101))                       # 1..100

    assert hb.percentile(data, 50) == 50
    assert hb.percentile(data, 95) == 95
    assert hb.percentile(data, 99) == 99
    assert hb.percentile(data, 100) == 100
    assert hb.percentile([7], 95) == 7


def test_percentile_kosong_error():
    with pytest.raises(ValueError, match="tidak ada sampel"):
        hb.percentile([], 95)


def test_summarise_mencatat_status_dan_reconnect():
    hasil = hb.summarise([1.0, 2.0, 3.0], Counter({200: 3}), reconnect=2)

    assert hasil["n"] == 3
    assert hasil["median_ms"] == 2.0
    assert hasil["status"] == {"200": 3}
    assert hasil["reconnect"] == 2


def test_summarise_tanpa_sampel_error():
    with pytest.raises(ValueError, match="tidak ada sampel"):
        hb.summarise([], Counter())


def test_galat_baku_median():
    assert hb.galat_baku_median(1.0, 100) == pytest.approx(0.12533, abs=1e-5)
    assert hb.galat_baku_median(0.0, 10) == 0.0
    with pytest.raises(ValueError, match="n harus >= 1"):
        hb.galat_baku_median(1.0, 0)


def test_bagi_ronde_totalnya_tepat_reps():
    for reps, rounds in ((60, 4), (7, 3), (4, 4), (10, 7)):
        bagian = hb._bagi_ronde(reps, rounds)
        assert len(bagian) == rounds
        assert sum(bagian) == reps
        assert min(bagian) >= 0


# --------------------------------------------------------------------------- #
# pengaman target
# --------------------------------------------------------------------------- #
def test_loopback_diizinkan_tanpa_flag():
    hb.assert_target_allowed("127.0.0.1", False, False, 100, 10)
    hb.assert_target_allowed("localhost", False, False, 100, 10)


def test_target_publik_ditolak_tanpa_izin():
    with pytest.raises(SystemExit, match="bukan loopback"):
        hb.assert_target_allowed("93.184.216.34", False, False, 10_000, 10)


def test_target_publik_ditolak_dengan_izin_separuh():
    with pytest.raises(SystemExit, match="bukan loopback"):
        hb.assert_target_allowed("93.184.216.34", True, False, 10_000, 10)
    with pytest.raises(SystemExit, match="bukan loopback"):
        hb.assert_target_allowed("93.184.216.34", False, True, 10_000, 10)


def test_target_publik_diizinkan_penuh(capsys):
    hb.assert_target_allowed("93.184.216.34", True, True, 10_000, 10)
    assert "izin" in capsys.readouterr().out


def test_anggaran_request_dibatasi():
    with pytest.raises(SystemExit, match="melebihi batas"):
        hb.assert_target_allowed("127.0.0.1", False, False, 5, 6)


def test_alamat_multicast_ditolak():
    with pytest.raises(SystemExit, match="multicast"):
        hb.assert_target_allowed("224.0.0.1", False, False, 10, 1)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("argv, pesan", [
    (["--n", "0"], "--n harus >= 1"),
    (["--reps", "0"], "--reps harus >= 1"),
    (["--reps", "3", "--rounds", "9"], "--rounds harus antara 1 dan --reps"),
    (["--warmup", "-1"], "--warmup tidak boleh negatif"),
])
def test_cli_menolak_masukan_tidak_masuk_akal(argv, pesan, capsys):
    with pytest.raises(SystemExit):
        hb.main(argv)
    assert pesan in capsys.readouterr().err
