"""The shared DnCNN spec and its per-framework implementations.

**This module imports NOTHING, and that is load-bearing.** ``distilaide/__init__.py`` resolves every
backend-owned name lazily so a machine without torch never imports it; an ``__init__`` re-exporting
the implementations would drag in torch AND keras AND tensorflow AND flax AND mlx the moment
anything under this package — including the dependency-free :mod:`~distilaide.dncnn.spec` — is
touched. Import an implementation from its own module, or through the lazy package root.
"""
