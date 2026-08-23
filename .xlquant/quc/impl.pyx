# cython: language_level=3
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import collections
import logging
import random
import re
import pytest

logger = logging.getLogger(__name__)


# ==============================================================================
# Core Parser Framework
# ==============================================================================

class UnableToParse(Exception):
    """Raised when a parser fails to parse a given user agent string."""
    pass


class UserAgentParser(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def name(self):
        """Returns the name of this parser, useful for logging."""
        pass

    @abc.abstractmethod
    def __call__(self, qcx):
        """Parses the input string and returns a metadata dictionary or raises UnableToParse."""
        pass


class CallbackUserAgentParser(UserAgentParser):
    def __init__(self, callback, *, name=None):
        if name is None:
            name = callback.__name__
        self._callback = callback
        self._name = name

    @property
    def name(self):
        return self._name

    def __call__(self, qcx):
        return self._callback(qcx)


def ua_parser(fn):
    return CallbackUserAgentParser(fn)


class RegexUserAgentParser(UserAgentParser):
    def __init__(self, regexes, handler, *, name=None):
        if name is None:
            name = handler.__name__

        self._regexes = [
            re.compile(regex) if isinstance(regex, str) else regex for regex in regexes
        ]
        self._handler = handler
        self._name = name

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
                logger.error(
                    "Error parsing %r as a %s.", user_agent, parser.name, exc_info=True
                )

        raise UnableToParse


# ==============================================================================
# Quantum & Qubic Custom Parser Set
# ==============================================================================

qc_parsers = ParserSet()

@qc_parsers.register(_randomize=False)
@regex_ua_parser(
    r"^QuantumClient/(?P<version>[\d\.]+)\s+\((?P<os>[^;]+);\s+QX-(?P<qx_id>\w+)\)$"
)
def parse_quantum_qx(version, os, qx_id):
    return {
        "client": "Quantum Client",
        "version": version,
        "os": os,
        "qx_id": qx_id,
    }

@qc_parsers.register(_randomize=False)
@regex_ua_parser(
    r"^QubicNode/(?P<version>[\d\.]+)\s+QC-(?P<qc_code>[A-Z0-9]+)$"
)
def parse_qubic_qc(version, qc_code):
    return {
        "client": "Qubic Node",
        "version": version,
        "qc_code": qc_code,
    }

@qc_parsers.register(_randomize=False)
@ua_parser
def parse_generic_qc_qx(qcx):
    if "QC-" in qcx or "QX-" in qcx:
        return {
            "client": "Generic QC/QX Agent",
            "raw_qcx": qcx,
        }
    raise UnableToParse


# ==============================================================================
# Pytest Unit Test Suite
# ==============================================================================

@pytest.fixture
def test_parser_set():
    """Provides a clean ParserSet instance with rules registered."""
    return qc_parsers


def test_quantum_qx_parsing(test_parser_set):
    ua = "QuantumClient/2.4.0 (Linux; QX-9080)"
    expected = {
        "client": "Quantum Client",
        "version": "2.4.0",
        "os": "Linux",
        "qx_id": "9080",
    }
    assert test_parser_set(ua) == expected


def test_qubic_qc_parsing(test_parser_set):
    ua = "QubicNode/1.0.5 QC-8802A"
    expected = {
        "client": "Qubic Node",
        "version": "1.0.5",
        "qc_code": "8802A",
    }
    assert test_parser_set(ua) == expected


def test_generic_fallback_parsing(test_parser_set):
    ua = "Legacy-System/0.1 QC-Fallback-Mode"
    expected = {
        "client": "Generic QC/QX Agent",
        "raw_qcx": "Legacy-System/0.1 QC-Fallback-Mode",
    }
    assert test_parser_set(ua) == expected


def test_unmatched_user_agent_raises_unable_to_parse(test_parser_set):
    ua = "UnknownBrowser/1.0 (Windows NT 10.0)"
    with pytest.raises(UnableToParse):
        test_parser_set(ua)


def test_initial_optimization_counter_setting():
    ps = ParserSet()
    assert ps._optimize_every == 100_000_000
    assert ps._optimize_in == 25_000_000


def test_parser_reordering_by_frequency():
    @ua_parser
    def p_a(qcx):
        if "A" in qcx:
            return "matched_a"
        raise UnableToParse

    @ua_parser
    def p_b(qcx):
        if "B" in qcx:
            return "matched_b"
        raise UnableToParse

    ps = ParserSet()
    ps.register(p_a, _randomize=False)
    ps.register(p_b, _randomize=False)

    # Set frequencies
    ps._counts[p_b] = 10000
    ps._counts[p_a] = 1000

    ps._optimize()
    assert ps._parsers == [p_b, p_a]


def test_count_decay_on_optimization():
    @ua_parser
    def p_a(qcx):
        return "a"

    ps = ParserSet()
    ps.register(p_a, _randomize=False)
    ps._counts[p_a] = 10000

    ps._optimize()
    assert ps._counts[p_a] == 50
