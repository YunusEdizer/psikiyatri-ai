"""Yerel/klinik ağ için kendinden-imzalı (self-signed) TLS sertifikası üretir.

NEDEN: Tarayıcının kamera/mikrofon API'si (``getUserMedia``) yalnızca
``localhost`` veya **HTTPS** üzerinde çalışır. Otomatik görüşme başka bir
bilgisayardan (danışan bilgisayarı) ağ üzerinden açılacaksa, sunucu HTTPS
sunmalıdır; aksi hâlde kamera hiç başlamaz.

KULLANIM:
    python scripts/generate_cert.py --host 192.168.1.50
    uvicorn psikiyatri_ai.api.main:app --host 0.0.0.0 --port 8443 \\
        --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem

Sonra tarayıcıda https://192.168.1.50:8443 açılır. Kendinden-imzalı
sertifika olduğundan tarayıcı bir kez güvenlik uyarısı gösterir; klinik iç
ağda "yine de devam et" ile geçilebilir (ya da sertifika cihazlara elle
güvenilir olarak eklenir).

> Kendinden-imzalı sertifika yalnızca kontrollü iç ağ/pilot içindir.
> Halka açık dağıtımda gerçek bir CA sertifikası (ör. Let's Encrypt)
> veya bir ters vekil (reverse proxy: Caddy/nginx) kullanılmalıdır.
"""

from __future__ import annotations

import argparse
import datetime
import ipaddress
from pathlib import Path


def generate(host: str, out_dir: Path, days: int = 825) -> tuple[Path, Path]:
    """Verilen host için cert.pem + key.pem üretir."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    out_dir.mkdir(parents=True, exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    # SAN: localhost + 127.0.0.1 + verilen host (IP ya da ad).
    alt_names: list[x509.GeneralName] = [
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
    ]
    try:
        alt_names.append(x509.IPAddress(ipaddress.ip_address(host)))
    except ValueError:
        alt_names.append(x509.DNSName(host))

    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Eurotech-1 (dev)"),
            x509.NameAttribute(NameOID.COMMON_NAME, host),
        ]
    )
    now = datetime.datetime.now(datetime.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=days))
        .add_extension(x509.SubjectAlternativeName(alt_names), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )

    key_path = out_dir / "key.pem"
    cert_path = out_dir / "cert.pem"
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return cert_path, key_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Kendinden-imzalı TLS sertifikası üret.")
    parser.add_argument(
        "--host",
        default="localhost",
        help="Sertifikanın kapsayacağı host (IP veya ad; ör. 192.168.1.50)",
    )
    parser.add_argument("--out", type=Path, default=Path("certs"), help="Çıktı dizini")
    args = parser.parse_args()

    cert_path, key_path = generate(args.host, args.out)
    print(f"Sertifika: {cert_path}")
    print(f"Anahtar  : {key_path}")
    print("\nHTTPS ile çalıştır:")
    print(
        f"  uvicorn psikiyatri_ai.api.main:app --host 0.0.0.0 --port 8443 "
        f"--ssl-keyfile {key_path} --ssl-certfile {cert_path}"
    )


if __name__ == "__main__":
    main()
