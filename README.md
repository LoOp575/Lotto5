# Lotto5

Eksperimen pipeline prediksi angka togel 4 digit dengan TensorFlow/Keras.

**Ini bukan alat ramalan.** Angka undian tidak punya pola yang bisa dipelajari
dari 26 baris data; skrip ini mengukur apakah sebuah model bisa mengalahkan
tebakan, dan hampir selalu jawabannya tidak. Perlakukan keluarannya sebagai
sanity check pipeline, bukan prediksi.

## Data

`Logo.csv` — TSV tanpa header, satu baris = satu undian, 7 angka 4 digit:

```text
4959	9219	9689	6086	4909	4106	5154
3286	8242	7079	0745	8021	6172	5695
```

* Angka dibaca sebagai **string** supaya nol di depan tidak hilang (`0745`).
* Baris yang kolomnya kurang/lebih dibuang dengan peringatan.
  Saat ini **baris 27 hanya berisi 4 angka** (`7301 8951 4105 5817`) sehingga
  belum dipakai — lengkapi 3 angka sisanya agar ikut terhitung.

## Menjalankan

```bash
python3 -m pip install -r requirements.txt

python3 Togel.py --dry-run        # validasi data, tanpa TensorFlow
python3 Togel.py --posisi 1       # latih + evaluasi
python3 Togel.py --help
```

Alur:

1. Fitur `X[t]` = dekomposisi digit undian ke-`t-1` (7 posisi × 4 digit × 10 nilai
   = 280 fitur biner).
2. Target `y[t]` = angka 4 digit pada `--posisi` undian ke-`t` (10 000 kelas).
3. Split **kronologis** (data lama untuk latih, terbaru untuk validasi), bukan acak.
4. Dilaporkan top-1/top-5 hit rate **dan** baseline kelas mayoritas.

## Tes

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m pytest tests -q
python3 -m ruff check Togel.py bench tests
```

## Folder lain

* `bench/` — uji latensi header nginx (lihat `bench/RESULTS.md`).
* `REVIEW.md` — audit kode dan status perbaikannya.
