#!/usr/bin/env python3
"""Prediksi satu posisi angka togel (4 digit, 10 000 kelas) dari undian sebelumnya.

Formulasi yang dipakai
----------------------
* Satu baris = satu undian, 7 angka 4 digit (``Logo.csv``, dipisah TAB, tanpa header).
* Target ``y`` = angka 4 digit pada **satu posisi** (``--posisi``, 1..7) dari
  undian ke-``t``  -> 10 000 kelas (0000..9999).
* Fitur ``X`` = dekomposisi digit dari undian ke-``t-1`` (7 posisi x 4 digit x
  10 nilai = 280 fitur biner). Undian ke-``t`` tidak pernah masuk fiturnya
  sendiri, jadi tidak ada kebocoran antar-baris.

Metrik yang dilaporkan: top-1 dan top-5 hit rate, dibandingkan **baseline**
(kelas paling sering di data latih). Untuk lotere, baseline biasanya menang;
angka di skrip ini bukan ramalan, hanya pengukuran apakah model mengalahkan
tebakan.

Contoh
------
    python3 Togel.py --dry-run          # cek data + bentuk fitur tanpa TensorFlow
    python3 Togel.py --posisi 1 --epochs 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N_POSISI = 7
N_DIGIT = 4
N_NILAI = 10
N_KELAS = 10 ** N_DIGIT          # 10 000
N_FITUR = N_POSISI * N_DIGIT * N_NILAI  # 280

DATA_DEFAULT = Path(__file__).with_name("Logo.csv")


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def muat_data(path: Path | str = DATA_DEFAULT) -> pd.DataFrame:
    """Baca Logo.csv sebagai string 4 digit, buang baris yang tidak lengkap.

    File dibaca dengan ``dtype=str`` supaya nol di depan tidak hilang
    (``"0745"`` tetap ``"0745"``, bukan ``745``).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"data tidak ditemukan: {path}\n"
            "Letakkan Logo.csv di folder yang sama dengan Togel.py, atau "
            "tunjuk lewat --data."
        )

    kolom = [f"posisi{i}" for i in range(1, N_POSISI + 1)]
    mentah = pd.read_csv(path, sep="\t", header=None, dtype=str, names=kolom)

    # baris dengan kolom kurang/lebih: jangan ditambal, cukup dilaporkan
    tidak_lengkap = mentah.index[mentah.isna().any(axis=1)].tolist()
    if tidak_lengkap:
        nomor = ", ".join(str(i + 1) for i in tidak_lengkap)
        print(f"[peringatan] baris {nomor} tidak punya {N_POSISI} angka "
              f"-> dibuang (lengkapi datanya di {path.name})", file=sys.stderr)
        mentah = mentah.dropna(how="any")

    if mentah.empty:
        raise ValueError(f"{path} tidak berisi satu pun baris lengkap")

    # validasi isi sel: wajib tepat 4 digit
    pola = rf"^\d{{{N_DIGIT}}}$"
    salah = ~mentah.apply(lambda s: s.str.fullmatch(pola)).all(axis=1)
    if salah.any():
        baris = ", ".join(str(i + 1) for i in mentah.index[salah])
        contoh = mentah.loc[salah].iloc[0].tolist()
        raise ValueError(
            f"baris {baris} berisi sel yang bukan {N_DIGIT} digit: {contoh}"
        )

    return mentah.reset_index(drop=True)


def ke_digit(df: pd.DataFrame) -> np.ndarray:
    """(n_baris, 7, 4) int 0..9."""
    angka = df.to_numpy(dtype=str)
    return np.stack(
        [np.frombuffer("".join(baris).encode(), dtype=np.uint8).reshape(-1, 4) - 48
         for baris in angka]
    ).astype(np.int16)


def one_hot_digit(digits: np.ndarray) -> np.ndarray:
    """(7, 4) digit -> vektor biner 280."""
    n = digits.size
    out = np.zeros(n * N_NILAI, dtype=np.float32)
    out[np.arange(n) * N_NILAI + digits.ravel()] = 1.0
    return out


def ke_kelas(angka4: str) -> int:
    """'0745' -> 745."""
    return int(angka4)


def buat_dataset(df: pd.DataFrame, posisi: int) -> tuple[np.ndarray, np.ndarray]:
    """Bangun (X, y) berpasangan: fitur = undian t-1, target = posisi p undian t."""
    if not 1 <= posisi <= N_POSISI:
        raise ValueError(f"--posisi harus 1..{N_POSISI}, bukan {posisi}")

    digits = ke_digit(df)
    angka = df.to_numpy(dtype=str)

    X = np.stack([one_hot_digit(digits[t - 1]) for t in range(1, len(df))])
    y = np.array([ke_kelas(baris[posisi - 1]) for baris in angka[1:]],
                 dtype=np.int64)
    return X, y


def bagi_data(X: np.ndarray, y: np.ndarray, test_size: float):
    """Split kronologis: data lama untuk latih, data terbaru untuk validasi.

    Untuk data deret waktu, split acak akan membocorkan masa depan ke masa lalu.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError("--test-size harus di antara 0 dan 1")
    n_val = max(1, int(round(len(X) * test_size)))
    if len(X) - n_val < 1:
        raise ValueError(
            f"hanya {len(X)} sampel; terlalu sedikit untuk split "
            f"{1 - test_size:.0%}/{test_size:.0%}"
        )
    potong = len(X) - n_val
    return (X[:potong], X[potong:], y[:potong], y[potong:])


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def bangun_model(n_fitur: int = N_FITUR, seed: int = 42):
    """Model kecil dengan 10 000 output logit. TensorFlow diimpor di sini saja."""
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_fitur,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(N_KELAS),          # logit, tanpa softmax
    ])


def top_k_akurasi(y_true: np.ndarray, skor: np.ndarray, k: int) -> float:
    """Hit rate top-k untuk multi-kelas (skor: (n_sampel, n_kelas))."""
    if len(y_true) == 0:
        return float("nan")
    k = min(k, skor.shape[1])
    teratas = np.argpartition(-skor, k - 1, axis=1)[:, :k]
    return float(np.mean([y in baris
                            for y, baris in zip(y_true, teratas, strict=True)]))


def baseline_mayoritas(y_train: np.ndarray, y_val: np.ndarray) -> tuple[int, float]:
    """Kelas paling sering di data latih + top-1-nya di data validasi."""
    nilai, hitung = np.unique(y_train, return_counts=True)
    kelas = int(nilai[np.argmax(hitung)])
    return kelas, float(np.mean(y_val == kelas))


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Prediksi satu posisi angka 4 digit")
    p.add_argument("--data", default=str(DATA_DEFAULT), help="path Logo.csv")
    p.add_argument("--posisi", type=int, default=1, help="kolom target (1..7)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true",
                   help="hanya muat data dan cetak bentuk fitur, tanpa TensorFlow")
    args = p.parse_args(argv)

    df = muat_data(args.data)
    X, y = buat_dataset(df, args.posisi)
    X_tr, X_val, y_tr, y_val = bagi_data(X, y, args.test_size)

    print(f"undian lengkap : {len(df)} baris")
    print(f"sampel         : {len(X)} (latih {len(X_tr)}, validasi {len(X_val)})")
    print(f"fitur          : {X.shape[1]} biner (7 posisi x 4 digit x 10)")
    print(f"target         : posisi {args.posisi}, {N_KELAS} kelas")

    kelas_base, akurasi_base = baseline_mayoritas(y_tr, y_val)
    print(f"baseline       : kelas {kelas_base:04d} "
          f"-> top-1 validasi {akurasi_base:.1%}")

    if args.dry_run:
        print("\n[dry-run] TensorFlow tidak diimpor; pipeline data sudah valid.")
        return 0

    if len(X) < 1000:
        print(f"\n[catatan] {len(X)} sampel untuk {N_KELAS} kelas: model tidak "
              "mungkin belajar pola nyata. Angka di bawah = sanity check "
              "pipeline, bukan ramalan.", file=sys.stderr)

    import tensorflow as tf

    model = bangun_model(X.shape[1], args.seed)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5")],
    )
    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=args.epochs, batch_size=args.batch_size, verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True)],
    )

    skor = model.predict(X_val, verbose=0)
    print(f"\ntop-1 validasi  : {top_k_akurasi(y_val, skor, 1):.2%} "
          f"(baseline {akurasi_base:.2%})")
    print(f"top-5 validasi  : {top_k_akurasi(y_val, skor, 5):.2%}")

    # prediksi undian berikutnya
    fitur_terakhir = one_hot_digit(ke_digit(df)[-1])[None, :]
    skor_baru = model.predict(fitur_terakhir, verbose=0)[0]
    urut = np.argsort(-skor_baru)[:5]
    print("\n5 kandidat teratas untuk undian berikutnya (posisi "
          f"{args.posisi}): " + ", ".join(f"{k:04d}" for k in urut))
    print("Peringatan: peluang satu nomor 4 digit = 1/10 000. Ini eksperimen "
          "pipeline, bukan prediksi yang bisa dipakai.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
