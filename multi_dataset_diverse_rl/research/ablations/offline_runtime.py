"""Fake provider adapter requires the guard before importing any provider."""
import socket

from ...search.schemas import SearchContractError
from .runtime import AblationBroker


class OfflineAblationBroker(AblationBroker):
    def __init__(self, **kwargs):
        if socket.socket.connect.__module__ != 'sitecustomize':
            raise SearchContractError('ABLATION_OFFLINE_NETWORK_GUARD_REQUIRED')
        super().__init__(**kwargs)
