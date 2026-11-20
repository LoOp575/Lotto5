# Audit kode — apa yang harus diperbaiki

Diperiksa: `Togel.py`, `Logo.csv`, `README.md`, dan `bench/` (harness yang saya
tambahkan). Semua klaim di bawah sudah dijalankan di sandbox, bukan dugaan.

Lingkungan uji: Python 3.11, pandas/scikit-learn/scipy terpasang di venv.
TensorFlow **tidak** dipasang (±600 MB), jadi baris Keras diverifikasi sampai
titik pembentukan input saja — bukan alasan untuk tidak memperbaiki, karena
skrip sudah mati jauh sebelum `model.compile`.

Ringkasan: **`Togel.py` tidak bisa dijalankan sama sekali** (4 kegagalan
berurutan), dan setelah dibuat jalan pun formulasi modelnya tidak bisa belajar
apa pun dari data yang ada.

---

## P0 — pemblokir: skrip berhenti sebelum melatih apa pun

### T1. File data tidak ada

`Togel.py:8` — `pd.read_csv('data_lottery.csv')`

```text
FileNotFoundError: [Errno 2] No such file or directory: 'data_lottery.csv'
```

Repo hanya punya `Logo.csv`. Perbaikan: baca `Logo.csv` (atau ganti nama file
data), dan pakai path relatif terhadap skrip, bukan cwd:

```python
DATA = Path(__file__).with_name("Logo.csv")
```

### T2. Skema kolom tidak cocok

`Togel.py:9-10` — `data['Numbers']`, `data['Result']`

```text
KeyError: 'Numbers'
```

`Logo.csv` adalah **TSV tanpa header**, 7 kolom angka 4 digit:

```text
4959\t9219\t9689\t6086\t4909\t4106\t5154
```

Dibaca dengan `pd.read_csv('Logo.csv')` (separator koma) seluruh baris jadi
**satu** kolom. Perbaikan: `pd.read_csv(DATA, sep="\t", header=None, dtype=str,
names=[f"n{i}" for i in range(1, 8)])`. Kolom `Result` memang tidak ada di
data — lihat T5.

### T3. `input_shape` diambil dari sparse matrix

`Togel.py:18` — `input_shape=(len(numbers_encoded[0]),)`

```text
TypeError: sparse array length is ambiguous; use getnnz() or shape[0]
```

Terjadi **apa pun** isi file datanya, jadi ini kegagalan yang tidak bisa
dihindari. Perbaikan: `input_shape=(numbers_encoded.shape[1],)`.

### T4. `predict` memakai bentuk dan skala yang salah

`Togel.py:33-34` — `np.array([[2, 4, 6, 8, 10]])` berukuran `(1, 5)`, sementara
input model `(186,)`. Selain itu angka mentah tidak pernah melewati
`OneHotEncoder` yang dipakai saat training, jadi artinya pun berbeda.
Perbaikan: `encoder.transform(...)` dengan bentuk yang sama seperti training.

### T5. Label/target tidak didefinisikan

Tidak ada kolom `Result`. Datanya 7 angka per undian, tetapi model punya **satu**
output `sigmoid` + `binary_crossentropy`. Tidak jelas apa yang diprediksi:
angka pertama? "menang/kalah"? Tanpa definisi target, `y_train` tidak bisa
dibentuk dan akurasi tidak punya arti. Ini keputusan desain yang harus
ditetapkan lebih dulu (mis. klasifikasi 4 digit pada satu posisi, atau
multi-output 7×10000).

---

## P1 — jalan, tapi hasilnya menyesatkan

### T6. One-hot pada angka 4 digit = matriks identitas

`Togel.py:13`. Terukur pada data yang ada:

| metrik | nilai |
| --- | --- |
| sampel | 186 |
| kolom one-hot | 186 |
| density | 0,0054 |

Setiap undian menjadi kategori uniknya sendiri (density ≈ 1/186 = vektor
one-hot), jadi model hanya menghafal indeks baris. Tidak ada generalisasi yang
mungkin. Perbaikan: vocabulary tetap (mis. satu-hot per posisi 0–9, atau
`OneHotEncoder(categories=[list(range(10000))], handle_unknown="ignore")`),
lalu bentuk input = 7 × 10 atau 7 × 10000.

### T7. Angka nol di depan hilang

`'0745'` → `745` (terbukti). Dua undian berbeda bisa jadi ID yang sama, dan
model kehilangan posisi digit. Perbaikan: baca sebagai `str`, jangan `int`.

### T8. Kebocoran preprocessing

`Togel.py:13-14`: encoder di-`fit_transform` pada **seluruh** data sebelum
`train_test_split`. Encoder harus `fit` di train saja, `transform` di val/test.

### T9. Tidak reproducible

Tidak ada `tf.keras.utils.set_random_seed(...)` / `np.random.seed`. `random_state=42`
hanya mengatur split, bukan inisialisasi bobot dan dropout.

### T10. Tidak ada baseline

Tidak ada pembanding (mis. "prediksi angka paling sering muncul"). Tanpa itu,
`Validation Accuracy` tidak bisa ditafsirkan — pada lotere, menebak satu angka
4 digit peluangnya 1/10000.

### T11. Loss/metrik tidak cocok dengan masalahnya

`binary_crossentropy` + `accuracy` untuk target yang bukan biner tunggal.
Untuk multi-label: `sigmoid` + `binary_crossentropy` per posisi memang bisa,
tapi `accuracy` harus diganti metrik per-posisi (mis. top-k hit rate).

### T12. Overfit hampir pasti, tanpa kendali

186 sampel, dua layer `Dense(64)` = ~16 ribu parameter, 10 epoch hardcoded,
tanpa `EarlyStopping`, `Dropout`, atau regularization. Tidak ada pencatatan
loss curve.

### T13. Struktur skrip

Tanpa `if __name__ == "__main__"`, tanpa penanganan file hilang, tanpa
`logging`, tanpa pemisahan fungsi (`load_data`/`build_model`/`evaluate`).
Akibatnya tidak bisa diimpor atau diuji.

---

## P2 — data & higiene repo

| # | Temuan | Bukti | Perbaikan |
| --- | --- | --- | --- |
| D1 | `Logo.csv` baris 27 hanya **4 kolom** (ragged) | `['7301','8951','4105','5817']` → NaN saat `read_csv`, dan `IntCastingNaNError` saat di-`astype(int)` | lengkapi jadi 7 angka atau hapus baris; loader wajib memvalidasi `shape` dan `dropna` |
| D2 | Dataset hanya 27 baris (26 lengkap), 186 angka unik semua | 27 × 7, tanpa duplikat | tidak cukup untuk melatih apa pun; dokumentasikan sumbernya di README |
| D3 | Tidak ada `requirements.txt` | README menyebut TF/Keras tapi tanpa versi | pin `tensorflow`, `pandas`, `numpy`, `scikit-learn` |
| D4 | Tidak ada LICENSE, CI, atau tes | `ls` | tambahkan minimal smoke test |
| D5 | Nama file tidak konsisten | commit `Loto.csv`, file `Logo.csv` | kemungkinan salah ketik ("Lotto"); putuskan satu nama |
| D6 | README satu baris | `README.md` | skema data + cara jalan + batasan (ini prediksi acak, bukan prediksi) |

---

## B — temuan pada harness `bench/header_bench.py` (kode saya)

| # | Temuan | Bukti | Perbaikan |
| --- | --- | --- | --- |
| B1 | `--n 0` dan `--reps 0` diterima diam-diam dan dikoersi | `--n 0` mencetak `n=0 (15 byte header)` padahal mengirim 1 header; `--reps 0` jadi 4 sampel | validasi `n >= 1`, `reps >= rounds` |
| B2 | Ambang "di atas derau" memakai konstanta 0,03 ms | `max(noise, 3 * 0.01)` — di mesin lambat 0,03 ms itu derau | `max(noise, 0.02 * base)` + ulang-ukur **kedua** varian |
| B3 | Penentuan koneksi ditutup memakai `resp.isclosed()` | `:119` — semantik "response selesai dibaca", bukan "server menutup" | pakai `resp.will_close`, dan laporkan jumlah reconnect |
| B4 | Warm-up memakai jalur berbeda di mode keepalive | `measure_variant` memanggil `one_shot` | warm-up lewat jalur yang sama dengan yang diukur |
| B5 | `self.method` di-set dua kali; `BODY` diabaikan untuk GET tanpa validasi | `:89-90` | rapikan; tolak `--method GET` + body |
| B6 | Tidak ada tes untuk harness | jalankan manual | tambah `bench/test_header_bench.py` (percentile, builder header, gate target) |
| B7 | `socket.gethostbyname` hanya IPv4 | `assert_target_allowed` | `getaddrinfo` + cek semua alamat |
| B8 | Pelanggaran lint kosmetik | ruff: `UP035`, 3× `E501` | `ruff check --fix` |

### Kesalahan di `bench/RESULTS.md` (dokumen saya sendiri)

1. §2b diberi judul "POST → 405", padahal angka di tabel itu diukur dengan
   **GET** dan `error_log warn` (status 200). Angka POST/405 adalah baris
   terpisah di §2a-instan 8080.
2. §1 menyebut "`256 koneksi × 300 request`", padahal `throughput()` aslinya
   300 koneksi untuk 300 request (satu koneksi per request).

---

## Urutan perbaikan yang saya sarankan

1. **T1–T5** dulu, karena tanpa itu tidak ada yang bisa dijalankan atau diukur.
   Butuh keputusan Anda untuk T5 (target apa yang mau diprediksi).
2. **T6–T9** supaya hasilnya bukan artefak (encoding, leading zero, seed, split).
3. **D1** cepat (satu baris data rusak), lalu **D3/D6**.
4. **B1–B6** untuk harness, plus dua koreksi `RESULTS.md`.
5. **T10–T13 / D2/D4/D5** bersifat peningkatan, bukan pemblokir.
