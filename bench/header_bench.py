#!/usr/bin/env python3
"""Bandingkan biaya parsing request untuk nama header "normal" vs nama yang
sengaja dibuat bertabrakan (``__`` / ``^~``), seperti pada test_nginx.py.

Perbedaan penting dibanding skrip asli:

* ``http.client.request(..., headers=list_of_tuples)`` tidak didukung dan
  langsung melempar ``AttributeError: 'tuple' object has no attribute 'lower'``
  (CPython >= 3.9). Di sini header dikirim lewat ``putheader()``.
* Koneksi bisa dipakai ulang (``--mode keepalive``); default-nya begitu, karena
  membuka koneksi baru per request membuat latensi didominasi handshake TCP
  (dan, untuk 443, TLS) sehingga selisih biaya header tenggelam.
* Kode status dicatat. Tanpa itu, benchmark terhadap nginx statis bisa "sukses"
  padahal yang diukur cuma halaman error 405/400.
* Persentil memakai nearest-rank yang benar (bukan ``lat[int(reps*0.95)]``).
* Ada warm-up, ronde yang diselang-seling, dan pengukuran ulang tiap varian
  sebagai batas derau (noise floor), supaya selisih kecil tidak dilaporkan
  sebagai efek.
* Pengaman: hanya mengizinkan target loopback. Target non-loopback butuh
  ``--allow-remote --i-own-this-host``; skrip ini menolak host yang bukan milik
  Anda.

Contoh:
    python3 bench/header_bench.py                     # nginx lokal :8080
    python3 bench/header_bench.py --n 100 300 1000 --json out.json
"""

from __future__ import annotations

import argparse
import http.client
import ipaddress
import json
import math
import socket
import ssl
import statistics
import time
from collections.abc import Callable, Iterable, Sequence
from itertools import product

HeaderList = list[tuple[str, str]]

DEFAULT_METHOD = "GET"
BODY = b"x" * 64
METODE_TANPA_BODY = {"GET", "HEAD", "DELETE", "OPTIONS", "TRACE"}


# --------------------------------------------------------------------------- #
# pembuat header
# --------------------------------------------------------------------------- #
def headers_normal(n: int, width: int = 0) -> HeaderList:
    """n nama header unik yang seluruhnya karakter valid (huruf, angka, '-')."""
    if n < 1:
        raise ValueError(f"n harus >= 1, bukan {n}")
    out: HeaderList = []
    for i in range(n):
        base = f"x-probe-{i}"
        if width > len(base):
            base += "-" * (width - len(base))
        out.append((base, "1"))
    return out


def headers_collision(n: int, width: int = 0) -> HeaderList:
    """n nama header dari alfabet {"__", "^~"}.

    Nama-nama ini tidak pernah muncul di tabel hash nginx: selain bukan header
    yang dikenal, keduanya mengandung karakter di luar tchar (``^``, ``~``) atau
    underscore, jadi nginx menandainya ``invalid_header``.
    """
    if n < 1:
        raise ValueError(f"n harus >= 1, bukan {n}")
    k = max(1, (n - 1).bit_length())  # 2**k >= n
    keys = ["".join(t) for t in product(("__", "^~"), repeat=k)][:n]
    return [(kunci + "^" * max(0, width - len(kunci)), "1") for kunci in keys]


def lebar_seragam(sizes: Sequence[int]) -> int:
    """Panjang nama yang membuat kedua varian memakai byte identik."""
    if not sizes:
        return 0
    n_maks = max(sizes)
    k = max(1, (n_maks - 1).bit_length())
    lebar_collision = 2 * k
    lebar_normal = max(len(f"x-probe-{n_maks - 1}"), 1)
    return max(lebar_collision, lebar_normal)


def header_bytes(hdrs: Sequence[tuple[str, str]]) -> int:
    """Perkiraan byte di wire: 'Nama: nilai\\r\\n' per header."""
    return sum(len(k) + len(v) + 4 for k, v in hdrs)


# --------------------------------------------------------------------------- #
# transport
# --------------------------------------------------------------------------- #
class Client:
    def __init__(self, host: str, port: int, tls: bool, insecure: bool,
                 timeout: float, method: str):
        self.host, self.port, self.tls = host, port, tls
        self.timeout, self.method = timeout, method.upper()
        if self.method not in METODE_TANPA_BODY and not BODY:
            raise ValueError(f"metode {self.method} butuh body")
        self.ctx: ssl.SSLContext | None = None
        if tls:
            self.ctx = ssl.create_default_context()
            if insecure:
                self.ctx.check_hostname = False
                self.ctx.verify_mode = ssl.CERT_NONE

    def open(self) -> http.client.HTTPConnection:
        if self.tls:
            assert self.ctx is not None
            return http.client.HTTPSConnection(
                self.host, self.port, timeout=self.timeout,
                context=self.ctx, server_hostname=self.host)
        return http.client.HTTPConnection(self.host, self.port,
                                          timeout=self.timeout)

    def exchange(self, conn: http.client.HTTPConnection,
                 hdrs: Sequence[tuple[str, str]]) -> tuple[int, int]:
        """Kirim satu request pada koneksi yang sudah terbuka."""
        conn.putrequest(self.method, "/", skip_accept_encoding=True)
        for name, value in hdrs:
            conn.putheader(name, value)
        conn.endheaders(None if self.method in METODE_TANPA_BODY else BODY)
        resp = conn.getresponse()
        payload = resp.read()
        if resp.will_close:        # server menutup koneksi: jangan dipakai lagi
            conn.close()
        return resp.status, len(payload)


def one_shot(client: Client, hdrs: Sequence[tuple[str, str]]) -> tuple[int, int]:
    conn = client.open()
    try:
        return client.exchange(conn, hdrs)
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# statistik
# --------------------------------------------------------------------------- #
def percentile(sorted_vals: Sequence[float], p: float) -> float:
    """Nearest-rank percentile (p dalam persen)."""
    if not sorted_vals:
        raise ValueError("tidak ada sampel")
    rank = math.ceil(p / 100.0 * len(sorted_vals)) - 1
    return sorted_vals[min(max(rank, 0), len(sorted_vals) - 1)]


def galat_baku_median(stdev: float, n: int) -> float:
    """Galat baku median untuk distribusi normal: 1,2533 * sd / sqrt(n)."""
    if n < 1:
        raise ValueError("n harus >= 1")
    return 1.2533 * stdev / math.sqrt(n)


def summarise(lat: list[float], statuses: dict[int, int],
              reconnect: int = 0) -> dict:
    if not lat:
        raise ValueError("tidak ada sampel terukur")
    lat_sorted = sorted(lat)
    return {
        "n": len(lat),
        "median_ms": round(statistics.median(lat_sorted), 3),
        "mean_ms": round(statistics.fmean(lat_sorted), 3),
        "p95_ms": round(percentile(lat_sorted, 95), 3),
        "p99_ms": round(percentile(lat_sorted, 99), 3),
        "stdev_ms": (round(statistics.pstdev(lat_sorted), 3)
                     if len(lat_sorted) > 1 else 0.0),
        "status": {str(k): v for k, v in sorted(statuses.items())},
        "reconnect": reconnect,
    }


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
def measure_variant(client: Client, maker: Callable[[], HeaderList],
                    mode: str, reps: int, warmup: int,
                    lat: list[float], statuses: dict[int, int]) -> int:
    """Ukur satu varian; kembalikan jumlah reconnect yang terjadi."""
    reconnect = 0
    hdrs = maker()

    if mode == "keepalive":
        conn = client.open()
        try:
            for _ in range(warmup):     # warm-up lewat jalur yang sama
                try:
                    client.exchange(conn, hdrs)
                except (http.client.HTTPException, OSError):
                    conn.close()
                    conn = client.open()
            for _ in range(reps):
                t0 = time.perf_counter()
                try:
                    status, _ = client.exchange(conn, hdrs)
                except (http.client.HTTPException, OSError):
                    # server menutup koneksi keep-alive: sambung ulang, jangan
                    # catat sampel yang tercemar waktu handshake
                    conn.close()
                    conn = client.open()
                    reconnect += 1
                    continue
                lat.append((time.perf_counter() - t0) * 1_000)
                statuses[status] = statuses.get(status, 0) + 1
        finally:
            conn.close()
    else:
        for _ in range(warmup):
            one_shot(client, hdrs)
        for _ in range(reps):
            hdrs = maker()
            t0 = time.perf_counter()
            status, _ = one_shot(client, hdrs)
            lat.append((time.perf_counter() - t0) * 1_000)
            statuses[status] = statuses.get(status, 0) + 1
    return reconnect


def _bagi_ronde(reps: int, rounds: int) -> list[int]:
    """Bagi reps ke rounds sedemikian rupa sehingga totalnya tepat reps."""
    dasar, sisa = divmod(reps, rounds)
    return [dasar + (1 if i < sisa else 0) for i in range(rounds)]


def run(client: Client, sizes: Iterable[int], mode: str, reps: int,
        rounds: int, warmup: int, width: int) -> list[dict]:
    results: list[dict] = []
    per_ronde = _bagi_ronde(reps, rounds)

    for n in sizes:
        varian: list[tuple[str, Callable[[], HeaderList]]] = [
            ("normal", lambda n=n: headers_normal(n, width)),
            ("collision", lambda n=n: headers_collision(n, width)),
            # tiap varian diukur dua kali: selisihnya = batas derau
            ("normal (ulang)", lambda n=n: headers_normal(n, width)),
            ("collision (ulang)", lambda n=n: headers_collision(n, width)),
        ]
        lat = {label: [] for label, _ in varian}
        st = {label: {} for label, _ in varian}
        rc = {label: 0 for label, _ in varian}

        for label, maker in varian:
            rc[label] += measure_variant(client, maker, mode, 0, warmup,
                                         lat[label], st[label])

        for j in range(rounds):
            for label, maker in varian:
                rc[label] += measure_variant(client, maker, mode, per_ronde[j],
                                             0, lat[label], st[label])

        kosong = [label for label, _ in varian if not lat[label]]
        if kosong:
            raise RuntimeError(f"tidak ada sampel terukur untuk {kosong}")

        row = {"n_headers": n,
               "header_bytes": header_bytes(varian[0][1]()),
               "variants": {}}
        for label, _ in varian:
            row["variants"][label] = summarise(lat[label], st[label], rc[label])
        results.append(row)

        dasar = row["variants"]["normal"]
        koll = row["variants"]["collision"]
        derau = max(abs(row["variants"]["normal (ulang)"]["median_ms"]
                        - dasar["median_ms"]),
                    abs(row["variants"]["collision (ulang)"]["median_ms"]
                        - koll["median_ms"]))
        delta = koll["median_ms"] - dasar["median_ms"]
        # ambang = 3 x galat baku selisih median (>= 3 sigma), minimal 2% dasar
        se_selisih = math.hypot(galat_baku_median(dasar["stdev_ms"], dasar["n"]),
                                galat_baku_median(koll["stdev_ms"], koll["n"]))
        ambang = max(3 * se_selisih, 0.02 * dasar["median_ms"])
        row["noise_floor_ms"] = round(derau, 3)
        row["se_diff_ms"] = round(se_selisih, 3)
        row["threshold_ms"] = round(ambang, 3)
        row["delta_collision_ms"] = round(delta, 3)
        row["delta_exceeds_noise"] = abs(delta) > ambang

        print(f"\nn={n}  ({row['header_bytes']} byte header, "
              f"{mode}, {reps} sampel/varian)")
        for label, _ in varian:
            s = row["variants"][label]
            status = ",".join(f"{k}x{v}" for k, v in s["status"].items())
            print(f"  {label:18} median={s['median_ms']:7.2f} ms  "
                  f"p95={s['p95_ms']:7.2f} ms  sd={s['stdev_ms']:6.2f}  "
                  f"n={s['n']}  status={status}"
                  + (f"  reconnect={s['reconnect']}" if s["reconnect"] else ""))
        print(f"  {'selisih':18} {delta:+.2f} ms  "
              f"(derau {derau:.2f} ms, galat baku selisih {se_selisih:.3f} ms, "
              f"ambang {ambang:.2f} ms) -> "
              f"{'DI ATAS derau' if row['delta_exceeds_noise'] else 'di dalam derau'}")
    return results


# --------------------------------------------------------------------------- #
# pengaman target
# --------------------------------------------------------------------------- #
def alamat_target(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    try:
        info = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise SystemExit(f"tidak bisa me-resolve {host!r}: {exc}") from exc
    alamat = {ipaddress.ip_address(item[4][0]) for item in info}
    if not alamat:
        raise SystemExit(f"{host!r} tidak punya alamat IP")
    return sorted(alamat, key=str)   # IPv4 dan IPv6 tidak bisa dibandingkan langsung


def assert_target_allowed(host: str, allow_remote: bool, owns_host: bool,
                          max_requests: int, projected: int) -> None:
    alamat = alamat_target(host)
    if any(ip.is_multicast or ip.is_unspecified for ip in alamat):
        raise SystemExit(f"{host} menunjuk ke alamat multicast/unspecified.")

    lokal = (host in {"localhost", "localhost.localdomain"}
             or all(ip.is_loopback for ip in alamat))
    if not lokal:
        if not (allow_remote and owns_host):
            raise SystemExit(
                f"menolak menembak {host} ({', '.join(map(str, alamat))}): "
                "bukan loopback.\n"
                "Benchmark header seperti ini adalah pengujian beban; ke host\n"
                "yang bukan milik Anda itu pemindaian tanpa izin. Kalau host\n"
                "itu memang milik Anda, tambahkan --allow-remote "
                "--i-own-this-host."
            )
        print(f"[izin] target non-loopback {host} "
              f"({', '.join(map(str, alamat))}) atas permintaan eksplisit.")

    if projected > max_requests:
        raise SystemExit(f"proyeksi {projected} request melebihi batas "
                         f"--max-requests {max_requests}. Kecilkan "
                         "--reps/--rounds/--n.")


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--tls", action="store_true", help="pakai HTTPS (SNI + verifikasi)")
    ap.add_argument("--insecure", action="store_true",
                    help="matikan verifikasi sertifikat")
    ap.add_argument("--mode", choices=("keepalive", "new-conn"), default="keepalive")
    ap.add_argument("--n", type=int, nargs="+", default=[100, 300, 1000],
                    help="jumlah header per request")
    ap.add_argument("--reps", type=int, default=60, help="sampel per varian")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--width", type=int, default=0,
                    help="panjang nama header disamakan (0 = otomatis)")
    ap.add_argument("--method", default=DEFAULT_METHOD)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--max-requests", type=int, default=5000)
    ap.add_argument("--allow-remote", action="store_true")
    ap.add_argument("--i-own-this-host", action="store_true")
    ap.add_argument("--json", metavar="PATH", help="tulis hasil mentah ke JSON")
    args = ap.parse_args(argv)

    # validasi masukan: tolak nilai yang tidak masuk akal, jangan dikoersi diam-diam
    if any(n < 1 for n in args.n):
        ap.error("--n harus >= 1")
    if args.reps < 1:
        ap.error("--reps harus >= 1")
    if not 1 <= args.rounds <= args.reps:
        ap.error("--rounds harus antara 1 dan --reps")
    if args.warmup < 0:
        ap.error("--warmup tidak boleh negatif")
    if args.width < 0:
        ap.error("--width tidak boleh negatif")
    if args.width == 0:
        args.width = lebar_seragam(args.n)

    projected = len(args.n) * 4 * (args.warmup + args.reps)
    assert_target_allowed(args.host, args.allow_remote, args.i_own_this_host,
                          args.max_requests, projected)

    client = Client(args.host, args.port, args.tls, args.insecure,
                    args.timeout, args.method)

    scheme = "https" if args.tls else "http"
    print(f"target      : {scheme}://{args.host}:{args.port}/")
    print(f"mode        : {args.mode} {args.method}, {args.reps} sampel/varian, "
          f"{args.rounds} ronde, {args.warmup} warm-up")
    print(f"panjang nama: {args.width} karakter (byte per grup identik)")

    try:
        status, _ = one_shot(client, headers_normal(1, args.width))
    except Exception as exc:                      # noqa: BLE001
        raise SystemExit(f"target tidak menjawab: {exc!r}") from exc
    print(f"probe awal  : status {status}"
          + ("  (bukan 2xx -> yang diukur halaman error)"
             if not 200 <= status < 300 else ""))

    results = run(client, args.n, args.mode, args.reps, args.rounds,
                  args.warmup, args.width)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"target": f"{scheme}://{args.host}:{args.port}",
                       "mode": args.mode, "results": results}, fh, indent=2)
        print(f"\nJSON -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
