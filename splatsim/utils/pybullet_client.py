"""Bind the bare ``pybullet`` module to one physics client.

The robot servers keep ``self.pybullet_client = pybullet`` (the MODULE) plus a
separate ``self._pb_client_id``. Any call made through that module without an
explicit ``physicsClientId`` silently goes to client 0, which is only correct
by luck — as soon as a process holds more than one connection, calls land on
the wrong simulation and fail in confusing, non-local ways.

Wrapping the module once makes every call explicit, so callers can pass this
around exactly like a ``pybullet_utils.bullet_client.BulletClient``.
"""

from __future__ import annotations

import pybullet as pb


class BulletClientShim:
    """Adapts ``pybullet`` + a client id to the BulletClient-style interface.

    ``physicsClientId`` is injected into every call that does not already
    specify one, so code written against a real ``BulletClient`` works
    unchanged against a server's raw-module handle.
    """

    __slots__ = ("_client_id",)

    def __init__(self, client_id: int):
        self._client_id = int(client_id)

    @property
    def client_id(self) -> int:
        return self._client_id

    def __getattr__(self, name):
        attr = getattr(pb, name)
        if not callable(attr):
            return attr

        def wrapped(*args, **kwargs):
            kwargs.setdefault("physicsClientId", self._client_id)
            return attr(*args, **kwargs)

        return wrapped


def as_client(obj, client_id: int | None = None):
    """Return a client-like object for ``obj``.

    Accepts an existing BulletClient (returned unchanged), or the bare
    ``pybullet`` module plus a ``client_id`` to bind it to.
    """
    if obj is pb or getattr(obj, "__name__", None) == "pybullet":
        if client_id is None:
            raise ValueError(
                "the bare pybullet module needs a client_id to bind to; "
                "pass the server's _pb_client_id")
        return BulletClientShim(client_id)
    return obj
