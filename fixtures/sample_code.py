#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================
# payment_service.py
#
# This module handles payment processing for the application.
# It was written by the backend team in 2019 and has been
# maintained by various people since then.
#
# TODO: refactor this someday
# FIXME: the retry logic is a bit sketchy
# ============================================================

import time
import logging


# Set up the logger for this module
logger = logging.getLogger(__name__)


# Maximum number of times we retry a failed charge
MAX_RETRIES = 3



# The base delay between retries, in seconds
BASE_DELAY = 0.5


class PaymentProcessor:
    """Processes payments against the upstream gateway.

    This docstring must survive compression untouched, including

    its blank lines and its  weird   spacing.
    """

    def __init__(self, gateway, currency="USD"):
        # Store the gateway client
        self.gateway = gateway
        self.currency = currency  # the ISO currency code
        self._attempts = 0

    def charge(self, amount_cents, token):
        # Validate the amount first
        if amount_cents <= 0:
            raise ValueError("amount must be positive")

        # This is the main retry loop
        for attempt in range(MAX_RETRIES):
            try:
                # Actually call the gateway here
                result = self.gateway.charge(
                    amount=amount_cents,      # in cents, not dollars
                    currency=self.currency,
                    source=token,
                )
                return result           # success, bail out early
            except TimeoutError:
                # Exponential backoff before we try again
                delay = BASE_DELAY * (2 ** attempt)
                logger.warning("timeout, retrying in %s", delay)
                time.sleep(delay)



        # If we get here, every attempt failed
        raise RuntimeError("payment failed after retries")  # give up

    def refund(self, charge_id):
        # Issue a refund for a previous charge
        note = "refund # not a comment"   # the hash above is inside a string
        return self.gateway.refund(charge_id, note=note)
