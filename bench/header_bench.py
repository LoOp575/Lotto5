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
* Ada warm-up, ronde yang diselang-seling, dan pengukuran ulang varian yang
  sama sebagai batas derau (noise floor), supaya selisih kecil tidak
  dilaporkan sebagai efek.
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
import sys
import time
from itertools import product
from typing import Callable, Iterable, Sequence

HeaderList = list[tuple[str, str]]

DEFAULT_METHOD = "GET"
BODY = b"x" * 64


# --------------------------------------------------------------------------- #
# pembuat header
# --------------------------------------------------------------------------- #
def headers_normal(n: int, width: int = 0) -> HeaderList:
    """n nama header unik yang seluruhnya karakter valid (huruf, angka, '-')."""
    n = max(1, n)
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
    n = max(1, n)
    k = max(1, (n - 1).bit_length())  # 2**k >= n
    keys = ["".join(t) for t in product(("__", "^~"), repeat=k)][:n]
    return [(k_ + "^" * max(0, width - len(k_)), "1") for k_ in keys]


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
        self.timeout, self.method = timeout, method
        self.method = method
        self.ctx: ssl.SSLContext | None = None
        if tls:
            self.ctx = ssl.create_default_context()
            if insecure:
                self.ctx.check_hostname = False
                self.ctx.verify_mode = ssl.CERT_NONE

    def _connect(self) -> http.client.HTTPConnection:
        if self.tls:
            assert self.ctx is not None
            return http.client.HTTPSConnection(
                self.host, self.port, timeout=self.timeout,
                context=self.ctx, server_hostname=self.host)
        return http.client.HTTPConnection(self.host, self.port,
                                          timeout=self.timeout)

    def open(self) -> http.client.HTTPConnection:
        return self._connect()

    def exchange(self, conn: http.client.HTTPConnection,
                 hdrs: Sequence[tuple[str, str]]) -> tuple[int, int]:
        """Kirim satu request pada koneksi yang sudah terbuka."""
        conn.putrequest(self.method, "/", skip_accept_encoding=True)
        for name, value in hdrs:
            conn.putheader(name, value)
        conn.endheaders(None if self.method == "GET" else BODY)
        resp = conn.getresponse()
        payload = resp.read()
        if not resp.isclosed():        # jaga koneksi tetap sinkron
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


def summarise(lat: list[float], statuses: dict[int, int]) -> dict:
    lat_sorted = sorted(lat)
    return {
        "n": len(lat),
        "median_ms": round(statistics.median(lat_sorted), 3),
        "mean_ms": round(statistics.fmean(lat_sorted), 3),
        "p95_ms": round(percentile(lat_sorted, 95), 3),
        "p99_ms": round(percentile(lat_sorted, 99), 3),
        "stdev_ms": round(statistics.pstdev(lat_sorted), 3) if len(lat_sorted) > 1 else 0.0,
        "status": {str(k): v for k, v in sorted(statuses.items())},
    }


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
def measure_variant(client: Client, maker: Callable[[], HeaderList],
                    mode: str, reps: int, warmup: int,
                    lat: list[float], statuses: dict[int, int]) -> None:
    for _ in range(warmup):
        one_shot(client, maker())       # buang halaman pertama / cache dingin

    if mode == "keepalive":
        hdrs = maker()
        conn = client.open()
        try:
            for _ in range(reps):
                t0 = time.perf_counter()
                try:
                    status, _ = client.exchange(conn, hdrs)
                except (http.client.HTTPException, OSError):
                    # server menutup koneksi keep-alive: sambung ulang, jangan
                    # catat sampel yang tercemar waktu handshake
                    conn.close()
                    conn = client.open()
                    continue
                lat.append((time.perf_counter() - t0) * 1_000)
                statuses[status] = statuses.get(status, 0) + 1
        finally:
            conn.close()
    else:
        for _ in range(reps):
            hdrs = maker()
            t0 = time.perf_counter()
            status, _ = one_shot(client, hdrs)
            lat.append((time.perf_counter() - t0) * 1_000)
            statuses[status] = statuses.get(status, 0) + 1


def run(client: Client, sizes: Iterable[int], mode: str, reps: int,
        rounds: int, warmup: int, width: int) -> list[dict]:
    results: list[dict] = []
    per_round = max(1, reps // rounds)

    for n in sizes:
        variants: list[tuple[str, Callable[[], HeaderList]]] = [
            ("normal", lambda n=n: headers_normal(n, width)),
            ("collision", lambda n=n: headers_collision(n, width)),
            # varian yang sama diukur dua kali: batas derau antar-ronde
            ("normal (ulang)", lambda n=n: headers_normal(n, width)),
        ]
        lat = {label: [] for label, _ in variants}
        st = {label: {} for label, _ in variants}

        for vlabel, maker in variants:      # warm-up per varian
            measure_variant(client, maker, mode, 0, warmup, lat[vlabel], st[vlabel])

        for _ in range(rounds):             # ronde diselang-seling
            for vlabel, maker in variants:
                measure_variant(client, maker, mode, per_round, 0,
                                lat[vlabel], st[vlabel])

        payload = header_bytes(variants[0][1]())
        row = {"n_headers": n, "header_bytes": payload, "variants": {}}
        for vlabel, _ in variants:
            row["variants"][vlabel] = summarise(lat[vlabel], st[vlabel])
        results.append(row)

        base = row["variants"]["normal"]["median_ms"]
        coll = row["variants"]["collision"]["median_ms"]
        repl = row["variants"]["normal (ulang)"]["median_ms"]
        noise = abs(repl - base)
        delta = coll - base
        row["noise_floor_ms"] = round(noise, 3)
        row["delta_collision_ms"] = round(delta, 3)
        row["delta_exceeds_noise"] = abs(delta) > max(noise, 3 * 0.01)

        print(f"\nn={n}  ({payload} byte header)")
        for vlabel, _ in variants:
            s = row["variants"][vlabel]
            print(f"  {vlabel:16} median={s['median_ms']:7.2f} ms  "
                  f"p95={s['p95_ms']:7.2f} ms  sd={s['stdev_ms']:6.2f}  "
                  f"status={','.join(f'{k}x{v}' for k, v in s['status'].items())}")
        print(f"  {'selisih':16} {delta:+.2f} ms  (derau ulang-ukur: {noise:.2f} ms)"
              f" -> {'DI ATAS derau' if row['delta_exceeds_noise'] else 'di dalam derau'}")
    return results


# --------------------------------------------------------------------------- #
# pengaman target
# --------------------------------------------------------------------------- #
def assert_target_allowed(host: str, allow_remote: bool, owns_host: bool,
                          max_requests: int, projected: int) -> None:
    try:
        ip = ipaddress.ip_address(socket.gethostbyname(host))
    except (socket.gaierror, ValueError) as exc:
        sys.exit(f"tidak bisa me-resolve {host!r}: {exc}")

    if ip.is_multicast or ip.is_unspecified:
        sys.exit(f"{ip} bukan target yang masuk akal.")

    local = ip.is_loopback or host in {"localhost", "localhost.localdomain"}
    if not local:
        if not (allow_remote and owns_host):
            sys.exit(
                f"menolak menembak {host} ({ip}): bukan loopback.\n"
                "Benchmark header seperti ini adalah pengujian beban; ke host\n"
                "yang bukan milik Anda itu pemindaian tanpa izin. Kalau host\n"
                "itu memang milik Anda, tambahkan --allow-remote "
                "--i-own-this-host."
            )
        print(f"[izin] target non-loopback {host} ({ip}) atas permintaan eksplisit.")

    if projected > max_requests:
        sys.exit(f"proyeksi {projected} request melebihi batas --max-requests "
                 f"{max_requests}. Kecilkan --reps/--rounds/--n.")


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--tls", action="store_true", help="pakai HTTPS (SNI + verifikasi)")
    ap.add_argument("--insecure", action="store_true", help="matikan verifikasi sertifikat")
    ap.add_argument("--mode", choices=("keepalive", "new-conn"), default="keepalive")
    ap.add_argument("--n", type=int, nargs="+", default=[100, 300, 1000],
                    help="jumlah header per request")
    ap.add_argument("--reps", type=int, default=60, help="sampel per varian")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--width", type=int, default=0,
                    help="panjang nama header disamakan (0 = apa adanya)")
    ap.add_argument("--method", default=DEFAULT_METHOD)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--max-requests", type=int, default=5000)
    ap.add_argument("--allow-remote", action="store_true")
    ap.add_argument("--i-own-this-host", action="store_true")
    ap.add_argument("--json", metavar="PATH", help="tulis hasil mentah ke JSON")
    args = ap.parse_args(argv)

    if args.n and args.width == 0:
        args.width = max(len(f"x-probe-{max(args.n) - 1}"),
                         max(len("".join(t)) for t in product(("__", "^~"),
                             repeat=max(1, (max(args.n) - 1).bit_length()))))

    projected = len(args.n) * 3 * (args.warmup + args.reps)
    assert_target_allowed(args.host, args.allow_remote, args.i_own_this_host,
                          args.max_requests, projected)

    client = Client(args.host, args.port, args.tls, args.insecure,
                    args.timeout, args.method)

    scheme = "https" if args.tls else "http"
    print(f"target      : {scheme}://{args.host}:{args.port}/")
    print(f"mode        : {args.mode}, {args.reps} sampel/varian, "
          f"{args.rounds} ronde, {args.warmup} warm-up")
    print(f"panjang nama: {args.width} karakter (byte per grup identik)")

    try:
        status, _ = one_shot(client, headers_normal(1, args.width))
    except Exception as exc:                      # noqa: BLE001
        sys.exit(f"target tidak menjawab: {exc!r}")
    print(f"probe awal  : status {status}"
          + ("  (perhatikan: bukan 2xx -> yang diukur halaman error)"
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
