# Copyright (c) ModelScope Contributors. All rights reserved.
"""External plugin that registers the elastic-relay callbacks and loss.

Usage::

    swift sft ... \
      --external_plugins /abs/path/to/swift/elastic_relay/plugin.py \
      --callbacks elastic_stop elastic_state

Importing this module mutates ``swift.callbacks.callbacks_map`` and ``swift.loss.loss_map``
so that the names above can be resolved by the trainer. It is safe to import repeatedly.
"""
from swift.callbacks.mapping import callbacks_map
from swift.elastic_relay.callbacks import ElasticStateCallback, ElasticStopCallback
from swift.elastic_relay.constants import (LOSS_NAME, STATE_CALLBACK_NAME, STOP_CALLBACK_NAME)
from swift.elastic_relay.loss import ElasticCrossEntropyLoss
from swift.loss.mapping import loss_map

callbacks_map.setdefault(STATE_CALLBACK_NAME, ElasticStateCallback)
callbacks_map.setdefault(STOP_CALLBACK_NAME, ElasticStopCallback)
loss_map.setdefault(LOSS_NAME, ElasticCrossEntropyLoss)
