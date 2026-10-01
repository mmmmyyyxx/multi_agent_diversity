"""Pre-import public-data-only transport guard; never grants model access."""
import hashlib
import http.client
import json
import os
import socket
import ssl
import urllib.request
from urllib.parse import urlsplit

_LOG = os.environ.get("PUBLIC_DATA_NETWORK_LOG")
_KNOWN_IPS = set()


def record(kind, **fields):
    if _LOG:
        with open(_LOG, "a", encoding="utf-8") as stream:
            stream.write(json.dumps({"kind": kind, **fields}, sort_keys=True) + "\n")


def allowed(host):
    host = str(host or "").lower().rstrip(".")
    return (host in {"huggingface.co", "raw.githubusercontent.com", "github.com", "api.github.com",
                    "pypi.org", "files.pythonhosted.org", "localhost", "127.0.0.1", "::1"}
            or host.endswith((".hf.co", ".huggingface.co", ".xethub.hf.co", ".githubusercontent.com")))


def require(host):
    if not allowed(host):
        record("blocked_non_data_transport", destination_sha256=hashlib.sha256(str(host).encode()).hexdigest())
        raise PermissionError("PUBLIC_DATA_NETWORK_ONLY")


_open = urllib.request.OpenerDirector.open
def open_url(self, request, *args, **kwargs):
    value = request.full_url if isinstance(request, urllib.request.Request) else request
    parsed = urlsplit(value)
    require(parsed.hostname)
    if parsed.scheme != "https" or (parsed.hostname == "huggingface.co" and not parsed.path.startswith("/datasets/")):
        record("blocked_non_data_transport", destination_sha256=hashlib.sha256(str(value).encode()).hexdigest())
        raise PermissionError("HF_DATASET_PATH_ONLY")
    return _open(self, request, *args, **kwargs)
urllib.request.OpenerDirector.open = open_url


_dns = socket.getaddrinfo
def getaddrinfo(host, *args, **kwargs):
    require(host)
    result = _dns(host, *args, **kwargs)
    _KNOWN_IPS.update(row[4][0] for row in result)
    return result
socket.getaddrinfo = getaddrinfo

_connect = socket.socket.connect
def connect(self, address):
    if isinstance(address, tuple) and address[0] not in _KNOWN_IPS and address[0] not in {"127.0.0.1", "::1"}:
        require(address[0])
    return _connect(self, address)
socket.socket.connect = connect

_wrap = ssl.SSLContext.wrap_socket
def wrap_socket(self, *args, **kwargs):
    require(kwargs.get("server_hostname"))
    return _wrap(self, *args, **kwargs)
ssl.SSLContext.wrap_socket = wrap_socket

_request = http.client.HTTPConnection.putrequest
def putrequest(self, method, url, *args, **kwargs):
    host = getattr(self, "_tunnel_host", None) or self.host
    require(host)
    parsed = urlsplit(url)
    if parsed.hostname:
        require(parsed.hostname)
        host = parsed.hostname
    path = parsed.path
    if host == "huggingface.co" and not path.startswith("/datasets/"):
        record("blocked_non_data_transport", destination_sha256=hashlib.sha256(path.encode()).hexdigest())
        raise PermissionError("HF_DATASET_PATH_ONLY")
    record("allowed_http_attempt", host=host, method=method)
    return _request(self, method, url, *args, **kwargs)
http.client.HTTPConnection.putrequest = putrequest
record("guard_bootstrap", llm_credentials_present=False)
