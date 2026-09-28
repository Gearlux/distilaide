"""The ONE spelling of ``import keras`` for this package.

Keras 3 reads ``KERAS_BACKEND`` at IMPORT time and its own default is ``tensorflow``, which
``distilaide[keras]`` does not install — Keras is an API, and the compute engine is the operator's
choice. ``recordstream.keras`` owns the ``os.environ.setdefault`` to the first engine actually
PRESENT, and it must run in the lowest layer that imports keras: an import sorter puts
``from recordstream.keras import RecordSequence`` ABOVE any first-party line, so a shim of our own
would lose the race and reintroduce ``ModuleNotFoundError: No module named 'tensorflow'``.

So every distilaide module imports keras THROUGH here, and never ``import keras`` directly.
"""

from recordstream.keras import RecordSequence, keras

__all__ = ["RecordSequence", "keras"]
