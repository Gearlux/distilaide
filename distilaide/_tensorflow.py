"""The ONE place distilaide imports TensorFlow — and the ``KERAS_BACKEND`` ordering that must precede it.

The twin of :mod:`distilaide._keras`, and it exists for the mirror image of that module's problem.
:mod:`recordstream.keras` defaults ``KERAS_BACKEND`` to the first backend actually INSTALLED, in the
order ``torch`` -> ``tensorflow`` -> ``jax``. That is the right answer for a Keras run — Keras 3 is
an API and any engine underneath will do — but it is the WRONG one here: on a machine carrying both
``distilaide[lightning]`` and ``distilaide[tensorflow]`` (the workspace's own setup), torch wins, and
:class:`~distilaide.tensorflow_restorer.TensorFlowRestorer` would then drive a torch-backed
``keras.Model`` with ``tf.GradientTape``, which cannot work.

Keras reads the variable ONCE, at import time, so the default has to be in place before ANY module
imports keras. Which is why this module hands over the ``keras`` handle **as well as** ``tf``: a
consumer importing both from their "natural" homes would write ``from distilaide._keras import
keras`` beside ``from distilaide._tensorflow import tf``, and isort sorts ``_keras`` ahead of
``_tensorflow`` — importing keras one line before the default that was supposed to steer it.
Re-exporting here removes the ordering hazard instead of documenting it.

``setdefault``, not an assignment: an explicit ``KERAS_BACKEND=jax`` still wins, and
``TensorFlowRestorer.initialize()`` then fails with a message naming the mismatch instead of running
on an engine its training loop cannot drive.

:class:`~recordstream.keras.RecordSequence` is re-exported for the SAME reason as ``keras``, and it
is not decoration: the default above lives in ``recordstream.keras``, so importing the batching
adapter from its own module is what pins the variable — and isort sorts a ``recordstream`` line into
the third-party group, ABOVE every ``distilaide`` line.
"""

import os

# BEFORE the imports below, and before anything else imports keras — see the module docstring
# for why this cannot live at the bottom of an import block.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import tensorflow as tf  # noqa: E402 - deliberately after the environment default above
from recordstream.keras import RecordSequence, keras, keras_backend  # noqa: E402

__all__ = ["RecordSequence", "keras", "keras_backend", "tf"]
