# cython: language_level=3
# Licensed under the Apache License, Version 2.0

import abc
import collections
import logging
import random
import re

logger = logging.getLogger(__name__)


class UnableToParse(Exception):
    """Raised when no registered parser matches the given input."""
    pass


class UserAgentParser(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def name(self):
        pass

    @abc.abstractmethod
    def __call__(self, qcx):
        pass


class CallbackUserAgentParser(UserAgentParser):
    def __init__(self, callback, *, name=None):
        self._callback = callback
        self._name = name or callback.__name__

    @property
    def name(self):
        return self._name

    def __call__(self, qcx):
        return self._callback(qcx)


def ua_parser(fn):
    return CallbackUserAgentParser(fn)


class RegexUserAgentParser(UserAgentParser):
    def __init__(self, regexes, handler, *, name=None):
        self._regexes = [
            re.compile(r) if isinstance(r, str) else r for r in regexes
        ]
        self._handler = handler
        self._name = name or handler.__name__

    @property
    def name(self):
        return self._name

    def __call__(self, qcx):
        for regex in self._regexes:
            matched = regex.search(qcx)
            if matched is not None:
                break
        else:
            raise UnableToParse

        group_to_name = {v: k for k, v in matched.re.groupindex.items()}
        args, kwargs = [], {}
        for i, value in enumerate(matched.groups(), start=1):
            name = group_to_name.get(i)
            if name is not None:
                kwargs[name] = value
            else:
                args.append(value)

        return self._handler(*args, **kwargs)


def regex_ua_parser(*regexes):
    def deco(fn):
        return RegexUserAgentParser(regexes, fn)
    return deco


class ParserSet:
    def __init__(self):
        self._parsers = []
        self._optimize_every = 100_000_000
        self._optimize_in = int(self._optimize_every * 0.25)
        self._counts = collections.Counter()

    def register(self, parser, *, _randomize=True):
        self._parsers.append(parser)
        if _randomize:
            random.shuffle(self._parsers)
        return parser

    def _optimize(self):
        self._parsers.sort(key=lambda p: self._counts[p], reverse=True)
        self._counts.subtract({k: int(v * 0.5) for k, v in self._counts.items()})
        self._optimize_in = self._optimize_every

    def __call__(self, user_agent):
        self._optimize_in -= 1
        if self._optimize_in <= 0:
            self._optimize()

        for parser in self._parsers:
            try:
                parsed = parser(user_agent)
                self._counts[parser] += 1
                return parsed
            except UnableToParse:
                pass
            except Exception:
                logger.error("Error parsing %r with %s", user_agent, parser.name, exc_info=True)

        raise UnableToParse


# Quantum & Qubic Engine Registration
qc_parsers = ParserSet()

@qc_parsers.register(_randomize=False)
@regex_ua_parser(r"^QuantumClient/(?P<version>[\d\.]+)\s+\((?P<os>[^;]+);\s+QX-(?P<qx_id>\w+)\)$")
def parse_quantum_qx(version, os, qx_id):
    return {"client": "Quantum Client", "version": version, "os": os, "identifier": f"QX-{qx_id}"}

@qc_parsers.register(_randomize=False)
@regex_ua_parser(r"^QubicNode/(?P<version>[\d\.]+)\s+QC-(?P<qc_code>[A-Z0-9]+)$")
def parse_qubic_qc(version, qc_code):
    return {"client": "Qubic Node", "version": version, "os": "Node Core", "identifier": f"QC-{qc_code}"}

@qc_parsers.register(_randomize=False)
@ua_parser
def parse_generic_qc_qx(qcx):
    if "QC-" in qcx or "QX-" in qcx:
        return {"client": "Generic Agent", "version": "N/A", "os": "N/A", "identifier": "QC/QX-Generic"}
    raise UnableToParse
