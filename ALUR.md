# Alur kerja kode — `Togel.py`

Ditulis dari penelusuran eksekusi nyata (bukan dari membaca kode saja): setiap
bentuk data di bawah adalah keluaran sebenarnya dari `Logo.csv` yang ada di repo.

## 1. Gambaran singkat

```
Logo.csv (27 baris, 7 kolom, TSV)
   │  muat_data()          validasi bentuk & isi, buang baris tak lengkap
   ▼
DataFrame (26 × 7)          semua sel string 4 digit
   │  ke_digit()           "0745" -> [0,7,4,5]
   ▼
ndarray (26 × 7 × 4)        int 0..9
   │  one_hot_digit()      tiap (posisi, digit) -> 10 slot
   ▼
vektor 280 biner           7 × 4 × 10
   │  buat_dataset()       pasangkan undian t-1 (fitur) dengan posisi p undian t (target)
   ▼
X (25 × 280), y (25,)      y = kelas 0..9999
   │  bagi_data()          split kronologis 20 / 5
   ▼
X_tr, X_val, y_tr, y_val
   │  bangun_model()       Dense 280->64->64->10000 (logit)
   ▼
skor (5 × 10000)           top-1 / top-5 hit rate vs baseline
   │
   ▼
5 kandidat untuk undian berikutnya
```

## 2. Tiap tahap: masukan, keluaran, dan **kenapa** begitu

| # | Fungsi | Masukan → Keluaran (nyata) | Alasan logika |
| --- | --- | --- | --- |
| 0 | `DATA_DEFAULT` | `Path(__file__).with_name("Logo.csv")` | Path relatif terhadap **skrip**, bukan `cwd`, supaya bisa dijalankan dari folder mana pun |
| 1 | `muat_data` | `Logo.csv` → `(26, 7)`, kolom `posisi1..7` | `dtype=str` supaya `"0745"` tidak jadi `745`; `sep="\t"`; baris ragged (baris 27) dibuang **dan dilaporkan**, tidak ditambal |
| 2 | `ke_digit` | `(26,7)` str → `(26,7,4)` int16 | Satu angka 4 digit diperlakukan sebagai **4 digit terpisah**; tanpa ini vocab jadi 10 000 kelas per posisi |
| 3 | `one_hot_digit` | `(7,4)` → `(280,)`, 28 nilai `1.0` | Indeks = `posisi*40 + digit_ke*10 + nilai`. Setiap kemungkinan (posisi, digit, nilai) punya slot sendiri, jadi urutan tidak ambigu |
| 4 | `buat_dataset` | `DataFrame` → `X (25,280)`, `y (25,)` | `X[t]` dari undian ke-`t-1`, `y[t]` dari undian ke-`t`: satu baris **tidak pernah** jadi fitur bagi targetnya sendiri. 26 baris → 25 pasangan karena undian terakhir hanya dipakai sebagai fitur |
| 5 | `bagi_data` | `(25)` → latih 20 / validasi 5 | Split **kronologis** (bukan acak): 5 sampel terakhir jadi validasi, supaya tidak ada informasi "masa depan" masuk ke pelatihan |
| 6 | `bangun_model` | 280 → 64 → 64 → 10 000 | Output **logit** (tanpa softmax) karena loss `from_logits=True` lebih stabil secara numerik |
| 7 | `top_k_akurasi` | `(5, 10000)` → skalar | `argpartition` untuk top-k, `strict=True` pada `zip` supaya panjang yang tidak cocok langsung gagal |
| 8 | `baseline_mayoritas` | `y_tr`, `y_val` → kelas + akurasi | Tanpa pembanding, akurasi tidak bisa ditafsirkan |
| 9 | prediksi | `ke_digit(df)[-1]` → `(1,280)` | Undian terakhir dipakai sebagai fitur untuk memprediksi undian berikutnya — konsisten dengan aturan lag di tahap 4 |

## 3. Contoh satu pasangan, dilacak angka per angka

Undian baris 1 dan 2:

```text
baris 1: 4959 9219 9689 6086 4909 4106 5154
baris 2: 3286 8242 7079 0745 8021 6172 5695
```

`X[0]` = hasil encoding **baris 1**, `y[0]` = posisi 1 **baris 2** = `3286`.

Indeks aktif pertama pada `X[0]`: `[4, 19, 25, 39, 49, 52, 61, 79]`, dan jika
didekode dengan rumus `posisi*40 + digit_ke*10 + nilai`:

| indeks | posisi | digit ke- | nilai | cocok dengan |
| --- | --- | --- | --- | --- |
| 4 | 1 | 0 | 4 | `4959` digit ke-0 |
| 19 | 1 | 1 | 9 | `4959` digit ke-1 |
| 25 | 1 | 2 | 5 | `4959` digit ke-2 |
| 39 | 1 | 3 | 9 | `4959` digit ke-3 |
| 49 | 2 | 0 | 9 | `9219` digit ke-0 |
| 52 | 2 | 1 | 2 | `9219` digit ke-1 |
| 61 | 2 | 2 | 1 | `9219` digit ke-2 |
| 79 | 2 | 3 | 9 | `9219` digit ke-3 |

Total 28 indeks aktif (7 posisi × 4 digit), sisa 252 slot bernilai 0.

## 4. Alur pelatihan & evaluasi

```text
          X_tr (20,280)                    X_val (5,280)
                │                                │
        Dense(64, relu)  ── 17 984 parameter     │
        Dropout(0.2)                             │
        Dense(64, relu)  ──  4 160 parameter     │
        Dense(10000)     ── 650 000 parameter    │
                │                                │
             y_tr (20,)                      y_val (5,)
        sparse_categorical_               hit rate top-1/top-5
        crossentropy(from_logits)
```

Angka nyata dari data sekarang:

| Ukuran | Nilai |
| --- | --- |
| sampel latih / validasi | 20 / 5 |
| parameter total | 672 144 |
| **parameter per sampel latih** | **33 607** |
| peluang acak top-1 | 0,01 % (1/10 000) |
| peluang acak top-5 | 0,05 % (5/10 000) |
| kelas di latih / validasi | 20 / 5 |
| **kelas yang beririsan** | **0** |

Baris terakhir itu penting: **tidak ada satu pun kelas yang muncul di latih dan
validasi**. Konsekuensinya, top-1 di validasi akan 0 % untuk model apa pun, dan
angka itu tidak mengukur kualitas model — hanya mengukur ketidakcukupan data.
Skrip sekarang mencetak peringatan itu secara eksplisit.

## 5. Jalur error (semua dijalankan, bukan diasumsikan)

| Situasi | Perilaku |
| --- | --- |
| file tidak ada | `FileNotFoundError: data tidak ditemukan: …` + petunjuk |
| tidak ada baris lengkap | `ValueError: … tidak berisi satu pun baris lengkap` |
| sel bukan 4 digit | `ValueError: baris 5 berisi sel yang bukan 4 digit: […]` |
| baris kurang kolom (baris 27) | dibuang + `[peringatan] baris 27 tidak punya 7 angka` |
| hanya 1 undian | `ValueError: butuh minimal 2 undian …` (dulu: `need at least one array to stack`) |
| `--posisi 0` atau `8` | `ValueError: --posisi harus 1..7, bukan 8` |
| `--test-size 1.5` | `ValueError: --test-size harus di antara 0 dan 1` |
| sampel terlalu sedikit untuk split | `ValueError: hanya 1 sampel; terlalu sedikit untuk split 80%/20%` |
| latih & validasi tidak berbagi kelas | `[peringatan] tidak ada satu pun kelas …` (baru) |

## 6. Tiga temuan dari penelusuran ini

1. **F1 — pesan error menyesatkan.** `muat_data` dengan 1 baris dulu lolos sampai
   `np.stack([])` dan muncul `ValueError: need at least one array to stack`.
   Sekarang diperiksa lebih awal. ✅ diperbaiki
2. **F2 — metrik tanpa makna dibiarkan tampil.** Dengan 25 sampel unik, latih dan
   validasi tidak berbagi kelas, jadi 0 % itu sifat data, bukan sifat model.
   Skrip sekarang memperingatkan dan menyarankan target lebih kasar (mis. satu
   digit = 10 kelas). ✅ diperbaiki
3. **F3 — cara memeriksa kronologi.** Membandingkan `max(y_tr)` dengan
   `min(y_val)` **salah** (nilai kelas bukan urutan waktu). Pemeriksaan yang benar
   lewat indeks baris, dan sudah diverifikasi:
   `X_val[0] == one_hot_digit(d[20])` → `True`, `y_val[0] == posisi1 baris 21`
   → `True (2483)`. Split-nya sudah kronologis; yang salah cuma cara cek saya
   waktu menelusuri.

## 7. Alur `bench/header_bench.py` (ringkas)

1. **Gerbang target** — `assert_target_allowed`: resolve semua alamat, tolak
   non-loopback kecuali `--allow-remote --i-own-this-host`, tolak multicast,
   batasi `--max-requests`.
2. **Probe** — satu request valid untuk memastikan target menjawab dan melihat
   kode status (memperingatkan kalau bukan 2xx).
3. **Ukur** — 4 varian (normal, collision, dan ulangan keduanya) × `--rounds`
   ronde diselang-seling, `--warmup` request dibuang.
4. **Statistik** — median/p95/p99 nearest-rank, sd, hitungan status & reconnect.
5. **Keputusan** — `|Δ median|` dibandingkan dengan `3 × galat baku selisih
   median` (minimal 2 % dari median dasar), bukan angka absolut.
