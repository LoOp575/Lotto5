# Uji "header collision" ala `test_nginx.py` — hasil dan koreksi

Ringkasan: **premisnya tidak berlaku untuk nginx.** Nama header seperti `__^~__`
tidak bisa menyebabkan tabrakan hash di nginx, dan pada konfigurasi default
header itu justru **dibuang** sebelum menyentuh struktur data apa pun.

Semua angka di bawah dari nginx 1.26.3 asli yang saya bangun dan jalankan di
sandbox, di `127.0.0.1`. Skrip tidak dijalankan ke `195.88.211.92` (lihat
bagian [Kenapa target itu tidak disentuh](#kenapa-target-itu-tidak-disentuh)).

---

## 1. Skrip aslinya gagal sebelum mengirim satu request pun

```text
File "/usr/lib/python3.11/http/client.py", line 1306, in _send_request
    header_names = frozenset(k.lower() for k in headers)
AttributeError: 'tuple' object has no attribute 'lower'
```

`HTTPConnection.request(..., headers=...)` hanya menerima `dict` (atau
`email.message.Message`), bukan `list[tuple]`. `hdrs_normal()` dan
`hdrs_collision()` mengembalikan list of tuples, jadi `bench()` dan
`throughput()` mati di iterasi pertama. Perbaikannya: pakai `dict` atau
`putrequest()` + `putheader()` seperti di `bench/header_bench.py`.

Bug lain yang tetap ada setelah itu diperbaiki:

| Masalah | Akibat |
| --- | --- |
| `HTTPConnection` ke port **443** | Bukan TLS. Ke server HTTPS, byte plaintext dianggap handshake rusak → yang diukur reset/timeout, bukan nginx. Perlu `HTTPSConnection` + SNI. |
| Koneksi baru tiap request | Latensi didominasi handshake TCP (+TLS). Di localhost selisihnya sudah 25% throughput; lewat internet (RTT puluhan–ratusan ms) efek header tidak akan terlihat sama sekali. |
| Kode status tidak pernah dicek | Terhadap server statis, semua request dijawab **405**, jadi yang di-"benchmark" adalah pembuatan halaman error. |
| `p95 = lat[int(reps*.95)]` | Untuk `reps=30` mengambil indeks 28 dari 30 (≈p93–p97), bukan p95 nearest-rank. Kecil, tapi sebaiknya benar. |
| Tanpa warm-up, tanpa selang-seling, tanpa ulang-ukur | Selisih 0,1 ms tidak bisa dibedakan dari derau mesin. |
| `throughput()` membuka 300 koneksi untuk 300 request | Menghabiskan port ephemeral / TIME_WAIT; yang terukur adalah laju pembukaan koneksi, bukan laju pemrosesan request. |

---

## 2. Hasil pengukuran nyata (nginx 1.26.3, lokal)

Nama header disamakan panjangnya (20 karakter) di kedua varian supaya **jumlah
byte identik** — kalau tidak, `hdrs_collision()` di n=1000 menghasilkan nama
20 karakter sementara `hdrs_normal()` 11–13 karakter, dan yang terukur cuma
perbedaan panjang request.

### 2a. Varian normal vs collision — keep-alive, `error_log warn`

`bench/header_bench.py --port 8091 --n 100 300 1000`

| n header | byte header | normal (median) | collision (median) | selisih | ambang 3σ |
| --- | --- | --- | --- | --- | --- |
| 100 | 2 500 | 0,26 ms | 0,24 ms | −0,01 ms | 0,05 ms |
| 300 | 7 500 | 0,48 ms | 0,48 ms | +0,00 ms | 0,03 ms |
| 1000 | 25 000 | 1,25 ms | 1,29 ms | +0,04 ms | 0,07 ms |

80 sampel per varian, tiap varian diukur dua kali (kolom "diulang" ada di
keluaran skrip). Ambang = 3 × galat baku selisih median,
`1,2533·sd/√n` per varian. Ketiga baris **di dalam derau**; tanda selisihnya pun
berubah-ubah antar-run (−, +, +), tanda khas derau.

Yang tumbuh linear adalah biaya mem-**parse byte** header (2 500 → 25 000 byte
= 0,26 → 1,25 ms), bukan biaya hashing nama.

### 2b. Koneksi baru per request (desain asli skrip), `GET` → 200

| n | normal | collision | selisih |
| --- | --- | --- | --- |
| 100 | 0,30 ms | 0,30 ms | +0,00 ms |
| 300 | 0,56 ms | 0,56 ms | −0,00 ms |
| 1000 | 1,32 ms | 1,33 ms | +0,01 ms |

Semua di dalam derau. Catatan: pengukuran pertama saya memakai `POST` ke server
statis, yang dijawab **405**; angkanya 0,38 / 0,67 / 1,64 ms untuk varian normal
dan 0,38 / 0,64 / 1,64 ms untuk collision — pola yang sama, karena yang diukur
adalah pembuatan halaman error.

Collision **lebih cepat 6%**: header-nya dibuang lebih awal (lihat §3), jadi
lebih sedikit pekerjaan. Arah efeknya justru berlawanan dengan hipotesis.

### 2c. Satu-satunya mekanisme yang benar-benar memperberat nginx: log

Konfigurasi identik, hanya `error_log` level yang berbeda (eksperimen manual,
30 request, satu koneksi per request):

| n | `warn`: valid / invalid | `info`: valid / invalid |
| --- | --- | --- |
| 100 | 0,37 / 0,34 ms (0,94×) | 0,35 / 0,65 ms (**1,84×**) |
| 300 | 0,67 / 0,67 ms (1,00×) | 1,19 / 1,61 ms (**1,36×**) |
| 1000 | 1,62 / 1,53 ms (0,94×) | 1,64 / 3,07 ms (**1,87×**) |

8 header tidak valid menghasilkan 8 baris log:

```text
[info] client sent invalid header line: "^~__^~: 1" while reading client request headers
```

Jadi kalau ada vektor di sini, itu **amplifikasi log I/O**, bukan hash. Mitigasi:
`error_log ... warn` (atau lebih tinggi), dan `limit_req` di level koneksi.

### 2d. Menambah `$http_*` tidak menolong hipotesis

Konfigurasi dengan `add_header X-Scan "$http_x_probe_99999";` (header yang tidak
pernah dikirim ⇒ nginx memindai seluruh `headers_in.headers` setiap request):

| n | tanpa scan | dengan scan |
| --- | --- | --- |
| 100 | 0,36 ms | 0,35 ms |
| 300 | 0,63 ms | 0,65 ms |
| 1000 | 1,71 ms | 1,68 ms |

Pemindaian linear itu ada di source, tapi biayanya tenggelam di bawah biaya
parsing 25 KB header — dan jumlah header dibatasi keras oleh buffer (§3).

### 2e. Kenapa desain "koneksi baru" menyesatkan

Loopback, tanpa RTT jaringan, `GET`, n=300:

| mode | throughput |
| --- | --- |
| koneksi baru tiap request | 1 814 req/s |
| keep-alive | 2 109 req/s |

Selisih 16% itu murni `socket()`/`connect()`; di jaringan nyata porsinya jauh
lebih besar karena satu RTT handshake bisa 20–50× biaya parsing header.

---

## 3. Kenapa "collision" tidak bisa bekerja pada nginx

Dari source nginx 1.26.3:

1. **Tabel hash header bersifat compile-time, isinya tetap 30 nama.**
   `ngx_http_init_headers_in_hash()` (`src/http/ngx_http.c:413`) membangun
   `ngx_hash_t` (`max_size = 512`, `bucket_size = 64`) dari array statis
   `ngx_http_headers_in` yang berisi 30 entri (`Host`, `Connection`, …).
   Nama dari klien **tidak pernah di-insert** ke tabel itu, jadi tidak ada
   collision yang bisa dikonstruksi dari luar.

2. **Header tak dikenal masuk ke list, bukan tabel hash.**
   `h = ngx_list_push(&r->headers_in.headers)` (`src/http/ngx_http_request.c:1492`)
   — append O(1) ke `ngx_list_t`. Setelah itu hanya ada satu
   `ngx_hash_find()` (`:1521`) terhadap tabel 30 entri di atas, yang hasilnya
   `NULL` untuk semua nama buatan kita.

3. **Karakter `_`, `^`, `~` membuat header ditandai invalid.**
   `src/http/ngx_http_parse.c:887` (`_` saat `underscores_in_headers off`),
   `:900` dan `:925` (karakter di luar charset token). Lalu, dengan
   `ignore_invalid_headers on` (default):
   `src/http/ngx_http_request.c:1479` → header **di-`continue`**, tidak pernah
   masuk list. Itulah sebabnya varian collision justru lebih murah di §2b.

4. **Batas keras ukuran header.** `client_header_buffer_size 1k` +
   `large_client_header_buffers 4 8k` ⇒ maksimum ~32 KB untuk request line +
   seluruh header. Pada nama 20 karakter itu sekitar 1 280 header; melewatinya
   dijawab `400`/`494` (`request_header_too_large`), bukan diproses lebih lama.

Kesimpulan: untuk nginx, keluarga serangan *hash-collision pada nama header*
tidak punya permukaan. Serangan tabrakan hash header itu nyata pada server yang
memasukkan nama header ke hash map saat runtime (Java `HttpHeaders`, beberapa
implementasi Node/Go lama, Python 2 `dict`). Python 3 sudah memakai SipHash
ber-key acak, jadi target Python juga tidak lagi rentan.

Vektor header yang **memang** relevan untuk nginx, kalau tujuannya DoS:
jumlah header besar (dibatasi 32 KB), nilai header sangat panjang (dibatasi
buffer yang sama), banyak header duplikat yang dikonsumsi modul tertentu,
`error_log info` + header invalid (log flooding), serta `$http_*`/`$cookie_*`
yang dipakai berulang di config (pemindaian list linear per evaluasi).

---

## 4. Kenapa target itu tidak disentuh

`test_nginx.py` adalah generator beban: 30×2×3 + 600 ≈ 780 request dengan
header adversarial ke satu host. Terhadap `195.88.211.92` — alamat yang tidak
ada hubungannya dengan repo ini (`Lotto5` cuma berisi CSV togel) dan yang
kepemilikannya tidak diketahui — itu pengujian beban tanpa izin, terlepas dari
niatnya. Karena itu saya menggantinya dengan nginx yang saya bangun sendiri.

`bench/header_bench.py` menerapkan aturan yang sama secara otomatis: target
loopback bebas, target non-loopback ditolak kecuali diberi
`--allow-remote --i-own-this-host`, plus batas `--max-requests`.

Kalau `195.88.211.92` memang milik Anda, jalankan dari mesin Anda sendiri
supaya RTT-nya realistis:

```bash
python3 bench/header_bench.py --host 195.88.211.92 --port 443 --tls \
    --allow-remote --i-own-this-host --mode keepalive --n 100 300 1000
```

---

## 5. Cara mereproduksi

```bash
# 1. bangun nginx (tanpa PCRE/zlib/OpenSSL)
curl -fsSL -o nginx.tar.gz \
  https://codeload.github.com/nginx/nginx/tar.gz/refs/tags/release-1.26.3
tar xzf nginx.tar.gz && cd nginx-release-1.26.3
./auto/configure --prefix=/tmp/nginx-inst \
    --without-http_rewrite_module --without-http_gzip_module
make -j"$(nproc)" && make install

# 2. jalankan target
mkdir -p /tmp/www /tmp/nginx-inst/logs && printf 'ok\n' > /tmp/www/index.html
/tmp/nginx-inst/sbin/nginx -c "$PWD/../bench/nginx-test.conf" -p /tmp/nginx-inst

# 3. ukur
python3 bench/header_bench.py --port 8080 --n 100 300 1000 --json hasil.json
```

Opsi berguna: `--mode new-conn` (meniru desain asli, untuk menunjukkan
besarnya bias handshake), `--tls`, `--width` (menyamakan panjang nama),
`--rounds`, `--reps`, `--max-requests`.

---

## 6. Rekomendasi konfigurasi (kalau tujuannya tahan beban)

```nginx
client_header_buffer_size 1k;        # jangan dinaikkan tanpa alasan
large_client_header_buffers 4 8k;    # batas atas total header
ignore_invalid_headers on;           # default, biarkan
underscores_in_headers off;          # default, biarkan
client_header_timeout 10s;           # lambat = putus
error_log /var/log/nginx/error.log warn;   # jangan 'info' di produksi
limit_conn_zone $binary_remote_addr zone=perip:10m;
limit_conn perip 20;
limit_req_zone $binary_remote_addr zone=reqip:10m rate=20r/s;
limit_req zone=reqip burst=40 nodelay;
```

---

## 7. Perubahan setelah audit

* Skrip diperbaiki: validasi masukan (`--n`/`--reps`/`--rounds`), warm-up lewat
  jalur yang diukur, `resp.will_close` + hitungan reconnect, dan ambang
  "di atas derau" diganti 3 × galat baku selisih median (bukan konstanta
  0,03 ms). Tiap varian sekarang diukur dua kali.
* Dua koreksi dokumen: §2b memang `GET` → 200 (bukan `POST` → 405), dan
  `throughput()` membuka **300** koneksi untuk 300 request.
* Tes: `tests/test_header_bench.py` (24 tes) — termasuk bug nyata yang
  ditemukan tes: `sorted()` campuran IPv4/IPv6 melempar `TypeError`.
