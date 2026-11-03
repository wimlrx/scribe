"""cleanup.py: prompt builder, guards and the Cleaner's fallbacks against a fake Ollama server on
127.0.0.1; the live cases in fixtures/cleanup_cases.json against the real local model.

The live tests are skipped unless SCRIBE_LIVE_CLEANUP=1 (they need Ollama and the configured model;
SCRIBE_CLEANUP_MODEL overrides the model):

    set SCRIBE_LIVE_CLEANUP=1
    .venv\\Scripts\\python.exe -m unittest tests.test_cleanup -v
"""
from __future__ import annotations

import copy
import http.server
import json
import logging
import os
import re
import socket
import statistics
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import requests

from scribe import cleanup, config, learn

FIXTURES = Path(__file__).parent / "fixtures"
CASES = json.loads((FIXTURES / "cleanup_cases.json").read_text(encoding="utf-8"))
LIVE = os.environ.get("SCRIBE_LIVE_CLEANUP") == "1"
TERMS = ["Wispr Flow", "Acme", "BlueSky", "DataBridge", "Claude", "Opus", "Help Desk", "Project Planner",
         "Morningstar", "Capital IQ", "Bloomberg", "Kira", "ORCA"]
SAMPLE = "um so I think we should move the meeting to Friday"


def settings(url: str = "http://127.0.0.1:9", **cleanup_cfg) -> dict:
    s = copy.deepcopy(config.DEFAULTS)
    s.update(owner_name="Alex Morgan", languages=["en", "af"], vocabulary=list(TERMS))
    s["ollama"]["url"] = url
    s["dictation"]["cleanup"].update(cleanup_cfg)
    return s


def evaluate(case: dict, output: str, used: bool | None = None) -> list[str]:
    """What the output gets wrong against one fixture case; [] is a pass."""
    fails: list[str] = []
    low = output.lower()
    lines = [ln for ln in output.split("\n") if ln.strip()]
    for s in case.get("must", []):
        if s not in output:
            fails.append(f"missing {s!r}")
    for s in case.get("must_ci", []):
        if s.lower() not in low:
            fails.append(f"missing {s!r} (any case)")
    for s in case.get("must_not", []):
        if s.lower() in low:
            fails.append(f"has {s!r}")
    for s in case.get("must_not_words", []):
        if re.search(r"(?<!\w)" + r"\s+".join(map(re.escape, s.split())) + r"(?!\w)", output, re.I):
            fails.append(f"has word {s!r}")
    for rx in case.get("regex", []):
        if not re.search(rx, output):
            fails.append(f"no match for /{rx}/")
    numbered = sum(1 for ln in lines if re.match(r"^\s*\d+[.)]\s", ln))
    bullets = sum(1 for ln in lines if re.match(r"^\s*[-•*]\s", ln))
    if "numbered" in case and numbered != case["numbered"]:
        fails.append(f"{numbered} numbered lines, want {case['numbered']}")
    if "bullets" in case and bullets != case["bullets"]:
        fails.append(f"{bullets} bullet lines, want {case['bullets']}")
    if case.get("lead_in") and (not lines or re.match(r"^\s*(?:\d+[.)]|[-•*])\s", lines[0])):
        fails.append("no lead-in line above the list")
    if "lines" in case and len(lines) != case["lines"]:
        fails.append(f"{len(lines)} lines, want {case['lines']}")
    for s, n in case.get("max_count", {}).items():
        if output.count(s) > n:
            fails.append(f"{s!r} {output.count(s)} times, want at most {n}")
    if "ends_with" in case and not output.rstrip().endswith(case["ends_with"]):
        fails.append(f"does not end with {case['ends_with']!r}")
    if case.get("used") is True and used is False:
        fails.append("model output rejected, input pasted")
    return fails


# ---------------------------------------------------------------- fake Ollama

class FakeOllama:
    """A local HTTP server that answers /api/chat from a script and records each request body."""

    def __init__(self):
        self.requests: list[dict] = []
        self.reply = lambda body: (200, {"message": {"role": "assistant", "content": "Cleaned text."},
                                         "done_reason": "stop"}, 0.0)
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                fake.requests.append(body)
                status, payload, delay = fake.reply(body)
                if delay:
                    time.sleep(delay)
                data = (payload if isinstance(payload, str) else json.dumps(payload)).encode("utf-8")
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except OSError:
                    pass                    # the client gave up (timeout test)

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def answer(self, content: str, status: int = 200, delay: float = 0.0, done_reason: str = "stop"):
        self.reply = lambda body: (status, {"message": {"role": "assistant", "content": content},
                                            "done_reason": done_reason}, delay)

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------- pure functions

class TestDefaults(unittest.TestCase):
    def test_settings_keys(self):
        c = config.DEFAULTS["dictation"]["cleanup"]
        self.assertTrue(c["enabled"])
        self.assertTrue(c["model"])
        self.assertGreater(c["timeout_s"], 0)
        self.assertEqual(c["keep_alive"], "30m")
        self.assertEqual(config.DEFAULTS["vocabulary"], ["Claude"])
        self.assertEqual((config.DEFAULTS["owner_name"], config.DEFAULTS["languages"]), ("", ["en"]))

    def test_cases_cover_every_behaviour(self):
        seen = {c["behaviour"] for c in CASES["cases"]}
        for b in ("repeats", "fillers", "self-correction", "lists", "punctuation and numbers", "proper nouns",
                  "no translation", "never answer", "negation"):
            self.assertIn(b, seen)
        self.assertEqual(len({c["id"] for c in CASES["cases"]}), len(CASES["cases"]))

    def test_fallback_cases_pass_on_their_own_input(self):
        """Where pasting Whisper's text is acceptable (used null), the input itself must pass."""
        for case in CASES["cases"]:
            if case.get("used") is None:
                with self.subTest(case["id"]):
                    relaxed = {k: v for k, v in case.items() if k not in ("ends_with",)}
                    self.assertEqual(evaluate(relaxed, case["input"]), [])

    def test_latency_input_is_about_forty_words(self):
        self.assertTrue(38 <= len(CASES["latency_input"].split()) <= 46)


class TestPrompt(unittest.TestCase):
    def test_shape(self):
        msgs = cleanup.build_messages("so  um\n we should\tgo", TERMS)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("Wispr Flow, Acme, BlueSky", msgs[0]["content"])
        self.assertIn("Never translate", msgs[0]["content"])
        self.assertEqual([m["role"] for m in msgs[1:]], ["user", "assistant"] * len(cleanup.EXAMPLES) + ["user"])
        self.assertEqual(msgs[-1]["content"], "<dictation>\nso um we should go\n</dictation>")

    def test_owner_and_languages_from_settings(self):
        plain = cleanup.build_messages("a b c d", TERMS)[0]["content"]
        self.assertTrue(plain.startswith("You tidy up dictation for the user, who speaks English. "), plain[:80])
        self.assertNotIn("Afrikaans", plain)
        mixed = cleanup.build_messages("a b c d", TERMS, None, "Alex Morgan", ["en", "af"])[0]["content"]
        self.assertTrue(mixed.startswith("You tidy up dictation for Alex, who speaks English and Afrikaans and "
                                         "often switches language mid-sentence."), mixed[:120])
        self.assertIn('"ek bedoel"', mixed)
        self.assertIn("and/en, the/die", mixed)
        two = cleanup.system_prompt(TERMS, "", ["en", "de"])
        self.assertIn("the user, who speaks English and German and often switches", two)
        self.assertNotIn("Afrikaans", two)
        for prompt in (plain, mixed, two):
            self.assertNotRegex(prompt, r"\{\w+\}")

    def test_prefix_is_stable(self):
        """Rules and examples do not depend on the dictation, so Ollama's prompt cache covers them."""
        a = cleanup.build_messages("first dictation here now", TERMS, [("Kiera", "Kira")])
        b = cleanup.build_messages("a second one entirely", TERMS)
        self.assertEqual(a[:-1], b[:-1])

    def test_pairs_go_in_the_last_message(self):
        msgs = cleanup.build_messages("ask kiera for it please", TERMS, [("Kiera", "Kira")])
        self.assertTrue(msgs[-1]["content"].startswith('Spelling for this dictation: write "Kira" where you hear "Kiera".'))
        self.assertNotIn("Kiera", msgs[0]["content"])

    def test_terms_deduped_and_empty(self):
        self.assertEqual(cleanup.clean_terms(["Kira", " kira ", "", "Help  Desk", None]), ["Kira", "Help Desk"])
        self.assertIn("Names and terms: none.", cleanup.build_messages("a b c d", [])[0]["content"])

    def test_examples_are_clean_outputs(self):
        """The worked examples pass the code's own guards (a guard that rejects them is miscalibrated)."""
        for ex in cleanup.EXAMPLES:
            pairs, said, clean = ex if len(ex) == 3 else ([], *ex)
            with self.subTest(said[:30]):
                self.assertIsNone(cleanup.check(said, clean, TERMS, pairs))

    def test_relevant_pairs(self):
        pairs = [("Kiera", "Kira"), ("blue sky", "BlueSky"), ("Orca", "ORCA"), ("x", "x"), ("bad",), ("", "y")]
        got = cleanup.relevant_pairs("ask KIERA about the Blue  Sky pack", pairs)
        self.assertEqual(got, [("Kiera", "Kira"), ("blue sky", "BlueSky")])
        self.assertEqual(cleanup.relevant_pairs("Kieran called", [("Kiera", "Kira")]), [])
        self.assertEqual(cleanup.relevant_pairs("Kiera", [("Kiera", "Kira"), ("kiera", "Kíra")]), [("Kiera", "Kira")])
        many = [(f"w{i}", f"W{i}") for i in range(60)]
        got = cleanup.relevant_pairs(" ".join(f"w{i}" for i in range(60)), many, limit=40)
        self.assertEqual(len(got), 40)
        self.assertEqual(got[0], ("w0", "W0"))


class TestTidy(unittest.TestCase):
    def test_strips_wrappers(self):
        self.assertEqual(cleanup.tidy("<think>\nhmm\n</think>\n\nSend it.", "send it"), "Send it.")
        self.assertEqual(cleanup.tidy("<dictation>\nSend it.\n</dictation>", "send it"), "Send it.")
        self.assertEqual(cleanup.tidy("```\nSend it.\n```", "send it"), "Send it.")
        self.assertEqual(cleanup.tidy('"Send it."', "send it"), "Send it.")
        self.assertEqual(cleanup.tidy('"Send it."', '"send it"'), '"Send it."')

    def test_line_ends(self):
        self.assertEqual(cleanup.tidy("Lead:  \r\n1. A\r\n\r\n\r\n\r\n2. B ", "x"), "Lead:\n1. A\n\n2. B")


class TestRepeats(unittest.TestCase):
    def test_collapse(self):
        said = ("I want to make it work the same as Whisper Flow in the sense that I want to make it work the "
                "same as Whisper Flow in the sense that I can give three bullets.")
        self.assertEqual(cleanup.collapse_repeats(said),
                         "I want to make it work the same as Whisper Flow in the sense that I can give three bullets.")
        self.assertEqual(cleanup.collapse_repeats("Send it to Adam. Send it to Adam. Today."), "Send it to Adam. Today.")
        self.assertEqual(cleanup.collapse_repeats("a b c d a b c d a b c d e"), "a b c d e")

    def test_short_and_spaced_repeats_kept(self):
        for said in ("very very good", "thank you thank you", "I said no, I said no", "the risk numbers and the risk numbers for Q3"):
            self.assertEqual(cleanup.collapse_repeats(said), said)

    def test_long_text_is_fast(self):
        long = " ".join(f"word{i % 97} other{i % 89}" for i in range(400))
        t0 = time.perf_counter()
        cleanup.collapse_repeats(long)
        self.assertLess(time.perf_counter() - t0, 0.5)

    def test_model_gets_the_collapsed_text(self):
        fake = FakeOllama()
        self.addCleanup(fake.close)
        fake.answer("Send the pack to Adam today.")
        c = cleanup.Cleaner(settings(fake.url, model="m", timeout_s=1.0))
        c.clean("send the pack to Adam send the pack to Adam today")
        self.assertIn("<dictation>\nsend the pack to Adam today\n</dictation>",
                      fake.requests[-1]["messages"][-1]["content"])


class TestRespell(unittest.TestCase):
    def test_soundex(self):
        self.assertEqual([cleanup.soundex(w) for w in ("Robert", "Rupert", "Ashcraft", "Tymczak", "Pfister", "")],
                         ["R163", "R163", "A261", "T522", "P236", ""])
        self.assertEqual(cleanup.soundex("whisper"), cleanup.soundex("Wispr"))

    def test_terms_any_case_and_spacing(self):
        got = cleanup.respell("finish the help desk clean-up for blue sky via data-bridge and morning star",
                              TERMS)
        self.assertEqual(got, "finish the Help Desk clean-up for BlueSky via DataBridge and Morningstar")
        self.assertEqual(cleanup.respell("the orca project", TERMS), "the ORCA project")
        self.assertEqual(cleanup.respell("Kira and kira", TERMS), "Kira and Kira")
        self.assertEqual(cleanup.respell("the morning start and Acmes", TERMS), "the morning start and Acmes")

    def test_sound_alike_multi_word_terms(self):
        self.assertEqual(cleanup.respell("like Whisper Flow and the whisper flow app", TERMS),
                         "like Wispr Flow and the Wispr Flow app")
        self.assertEqual(cleanup.respell("the project planer", TERMS), "the Project Planner")
        self.assertEqual(cleanup.respell("whisper slow and a flow", TERMS), "whisper slow and a flow")

    def test_learned_terms_are_hints_only(self):
        """A learned term from terms_fn reaches the prompt but is never forced on the model's text."""
        fake = FakeOllama()
        self.addCleanup(fake.close)
        fake.answer("Please mark the book for Wispr flow review.")
        c = cleanup.Cleaner(settings(fake.url, model="m", timeout_s=1.0),
                            terms_fn=lambda: list(TERMS) + ["Mark"],
                            pairs_fn=lambda: [("Kiera", "Kira")])
        self.assertEqual(c.clean("please mark the book for wispr flow review"),
                         "Please mark the book for Wispr Flow review.")
        self.assertIn("Mark.", fake.requests[-1]["messages"][0]["content"])


class TestGuards(unittest.TestCase):
    def ok(self, src, out, terms=TERMS, pairs=None):
        self.assertIsNone(cleanup.check(src, out, terms, pairs), out)

    def rejects(self, name, src, out, terms=TERMS, pairs=None):
        self.assertEqual(cleanup.check(src, out, terms, pairs), name, out)

    def test_clean_outputs_pass(self):
        self.ok("um send it to adam on tuesday no wait wednesday", "Send it to Adam on Wednesday.")
        self.ok("I want to list the following three points first the fund returned four and a half percent "
                "second costs went up third we need a new client",
                "I want to list the following three points:\n1. The fund returned 4.5%.\n2. Costs went up.\n"
                "3. We need a new client.")
        self.ok("it works like whisper flow and blue sky", "It works like Wispr Flow and BlueSky.")
        self.ok("ask kiera for the deck please", "Ask Kira for the deck, please.", pairs=[("Kiera", "Kira")])
        self.ok("I will call him at ten", "I'll call him at 10.")
        self.ok("the fee is twenty five thousand rand", "The fee is R25,000.")

    def test_empty(self):
        self.rejects("empty", "send the pack to adam", "  \n ")

    def test_preface(self):
        for out in ("Here is the cleaned text: Send the pack to Adam.", "Sure! Send the pack to Adam.",
                    "Cleaned: Send the pack to Adam.", "**Here's** the text: Send the pack to Adam."):
            self.rejects("preface", "send the pack to adam", out)
        self.ok("here is the deck for Monday's review", "Here is the deck for Monday's review.")
        self.ok("sure I can do the call on Friday", "Sure, I can do the call on Friday.")

    def test_length(self):
        src = "send the pack to adam before friday please"
        self.rejects("length", src, src + " " + " ".join(["and"] * 15))
        long_src = " ".join(["the client wants the numbers by friday"] * 4)
        self.rejects("length", long_src, "Friday.")
        self.ok("twenty five thousand rand", "R25,000")

    def test_added_content(self):
        self.rejects("added", "write me an email to adam about the pack",
                     "Dear Adam, please find the pack attached. Kind regards, Alex")
        self.assertIn(cleanup.check("what is the capital of france", "What is the capital of France? The capital "
                                    "of France is Paris, a city on the Seine.", TERMS), ("length", "added"))
        self.rejects("added", "can you tell me what the capital of france is before the meeting",
                     "Can you tell me what the capital of France is before the meeting? It is Paris.")
        self.rejects("added", "send the pack to adam before the meeting on friday",
                     "Send the pack to Adam before the meeting on Friday. Hope that helps!")
        self.ok("write me an email to adam about the pack", "Write me an email to Adam about the pack.")
        self.ok("the cache flows look fine this quarter for the fund", "The cash flows look fine this quarter for the fund.")

    def test_translation(self):
        self.rejects("translation", "ek sal dit dinsdag stuur nee ek bedoel donderdag", "I will send it on Thursday.")
        # "eers" came back as "first", an unsaid ordinal: the figures guard, which runs first, names it.
        self.assertIn(cleanup.check("ja so ons moet eers die numbers check and then we send it to the client",
                                    "Yes, so we must first check the numbers, and then we send it to the client.",
                                    TERMS), ("figures", "translation"))
        self.rejects("translation", "ja so ons moet eers die numbers check and then we send it to the client",
                     "Ja, so ons moet eers die numbers check, en dan we send it to die client.")
        self.rejects("translation", "please send the report to Adam before the meeting",
                     "Stuur asseblief die verslag vir Adam voor die vergadering.")
        self.ok("ek sal dit dinsdag stuur nee ek bedoel donderdag", "Ek sal dit Donderdag stuur.")
        self.ok("send it tuesday nee ek bedoel wednesday", "Send it Wednesday.")
        self.ok("the fry dug market opens at nine", "The Vrydag market opens at 9.", pairs=[("fry dug", "Vrydag")])
        self.ok("ja so ons moet eers die numbers check and then we send it to the client",
                "Ja, so ons moet eers die numbers check, and then we send it to the client.")


# ---------------------------------------------------------------- the Cleaner against a fake server

class TestCleaner(unittest.TestCase):
    def setUp(self):
        self.fake = FakeOllama()
        self.addCleanup(self.fake.close)
        self.s = settings(self.fake.url, model="test-model", timeout_s=1.0, keep_alive="30m")

    def test_ok(self):
        self.fake.answer("So I think we should move the meeting to Friday.")
        c = cleanup.Cleaner(self.s)
        self.assertEqual(c.clean(SAMPLE), "So I think we should move the meeting to Friday.")
        self.assertEqual(set(c.last), {"ms", "used", "reason"})
        self.assertEqual((c.last["used"], c.last["reason"]), (True, "ok"))
        body = self.fake.requests[-1]
        self.assertEqual(body["model"], "test-model")
        self.assertIs(body["think"], False)
        self.assertIs(body["stream"], False)
        self.assertEqual(body["keep_alive"], "30m")
        self.assertEqual(body["options"]["temperature"], 0)
        self.assertEqual(body["options"]["num_ctx"], cleanup.NUM_CTX)
        n = len(SAMPLE.split())
        self.assertEqual(body["options"]["num_predict"], cleanup.PREDICT_BASE + cleanup.PREDICT_PER_WORD * n)
        self.assertIn(SAMPLE, body["messages"][-1]["content"])

    def test_disabled_and_short_make_no_request(self):
        c = cleanup.Cleaner(settings(self.fake.url, enabled=False, model="m"))
        self.assertEqual(c.clean(SAMPLE), SAMPLE)
        self.assertEqual(c.last["reason"], "disabled")
        c = cleanup.Cleaner(self.s)
        self.assertEqual(c.clean("send it now"), "send it now")
        self.assertEqual((c.last["reason"], c.last["used"]), ("short", False))
        self.assertEqual(c.clean(""), "")
        self.assertIsNone(c.clean(None))
        self.assertEqual(self.fake.requests, [])

    def test_timeout_returns_input_quickly(self):
        self.fake.answer("Too late.", delay=1.5)
        c = cleanup.Cleaner(settings(self.fake.url, model="m", timeout_s=0.3))
        t0 = time.perf_counter()
        self.assertEqual(c.clean(SAMPLE), SAMPLE)
        self.assertLess(time.perf_counter() - t0, 1.0)
        self.assertEqual((c.last["reason"], c.last["used"]), ("timeout", False))

    def test_timeout_scales_with_length(self):
        c = cleanup.Cleaner(settings(self.fake.url, model="m", timeout_s=2.5, timeout_per_word_s=0.01))
        self.assertEqual(c.timeout_for(30), 2.5)
        self.assertAlmostEqual(c.timeout_for(140), 3.5)

    def test_errors_return_input(self):
        for status, payload in ((500, {"error": "boom"}), (404, {"error": "model not found"}), (200, "not json")):
            self.fake.reply = lambda body, st=status, pl=payload: (st, pl, 0)
            c = cleanup.Cleaner(self.s)
            with self.subTest(status=status):
                self.assertEqual(c.clean(SAMPLE), SAMPLE)
                self.assertEqual(c.last["reason"], "error")
        c = cleanup.Cleaner(settings(f"http://127.0.0.1:{free_port()}", model="m", timeout_s=1.0))
        self.assertEqual(c.clean(SAMPLE), SAMPLE)
        self.assertEqual(c.last["reason"], "error")

    def test_guard_rejection_returns_input(self):
        self.fake.answer("Sure! Here is your text: So I think we should move the meeting to Friday.")
        c = cleanup.Cleaner(self.s)
        self.assertEqual(c.clean(SAMPLE), SAMPLE)
        self.assertEqual((c.last["reason"], c.last["used"]), ("guard:preface", False))
        self.fake.answer("So I think we should move the meeting to Friday", done_reason="length")
        self.assertEqual(c.clean(SAMPLE), SAMPLE)
        self.assertEqual(c.last["reason"], "guard:truncated")

    def test_think_field_refused_is_dropped(self):
        def reply(body):
            if "think" in body:
                return 400, {"error": '"m" does not support thinking'}, 0
            return 200, {"message": {"content": "So I think we should move the meeting to Friday."},
                         "done_reason": "stop"}, 0
        self.fake.reply = reply
        c = cleanup.Cleaner(self.s)
        self.assertEqual(c.clean(SAMPLE), "So I think we should move the meeting to Friday.")
        self.assertEqual(c.clean(SAMPLE), "So I think we should move the meeting to Friday.")
        self.assertEqual(["think" in b for b in self.fake.requests], [True, False, False])

    def test_terms_and_pairs_callbacks(self):
        self.fake.answer("Ask Kira for the Zebra Fund pack.")
        c = cleanup.Cleaner(self.s, terms_fn=lambda: ["Zebra Fund"], pairs_fn=lambda: [("Kiera", "Kira")])
        self.assertEqual(c.clean("ask kiera for the zebra fund pack"), "Ask Kira for the Zebra Fund pack.")
        msgs = self.fake.requests[-1]["messages"]
        self.assertIn("Names and terms: Zebra Fund.", msgs[0]["content"])
        self.assertIn('write "Kira" where you hear "Kiera"', msgs[-1]["content"])

    def test_default_terms_are_the_vocabulary(self):
        self.fake.answer("So I think we should move the meeting to Friday.")
        cleanup.Cleaner(self.s).clean(SAMPLE)
        self.assertIn("Wispr Flow", self.fake.requests[-1]["messages"][0]["content"])

    def test_callback_failure_never_raises(self):
        c = cleanup.Cleaner(self.s, terms_fn=lambda: 1 / 0)
        self.assertEqual(c.clean(SAMPLE), SAMPLE)
        self.assertEqual(c.last["reason"], "error")

    def test_no_dictated_text_in_logs(self):
        secret = "the Zanzibar acquisition closes on Friday at noon"
        self.fake.answer("The Zanzibar acquisition closes on Friday at noon.")
        c = cleanup.Cleaner(self.s)
        with self.assertLogs("scribe.cleanup", level=logging.DEBUG) as cm:
            c.clean(secret)
            self.fake.answer("Sure, here: The Zanzibar acquisition closes Friday.")
            c.clean(secret)
            c.warm(wait=True)
        self.assertFalse(any("Zanzibar" in line for line in cm.output), cm.output)

    def test_warm(self):
        c = cleanup.Cleaner(self.s)
        c.warm(wait=True)
        body = self.fake.requests[-1]
        self.assertEqual(body["options"]["num_predict"], 1)
        self.assertEqual(body["keep_alive"], "30m")
        self.assertEqual(body["options"]["num_ctx"], cleanup.NUM_CTX)
        self.assertEqual(body["messages"][:-1],
                         cleanup.build_messages("x", c._terms(), None, "Alex Morgan", ["en", "af"])[:-1])

    def test_warm_is_single_flight_and_never_raises(self):
        self.fake.answer("x", delay=0.4)
        c = cleanup.Cleaner(self.s)
        c.warm()
        c.warm()
        time.sleep(0.6)
        self.assertEqual(len(self.fake.requests), 1)
        down = cleanup.Cleaner(settings(f"http://127.0.0.1:{free_port()}", model="m"))
        down.warm(wait=True)
        cleanup.Cleaner(settings(self.fake.url, enabled=False, model="m")).warm(wait=True)
        self.assertEqual(len(self.fake.requests), 1)


# ---------------------------------------------------------------- review fixes (31 Oct 2026)

class TestRespellNeedsCloseSpelling(unittest.TestCase):
    """F1: the sound-alike swap no longer turns ordinary finance words into product names."""

    def test_ordinary_phrases_stay(self):
        for said in ("the working capital is tight this quarter", "we ran the project planning on the fund",
                     "a capital issue", "project plainly weighted", "both help desks", "the help deck",
                     "the helpful desk"):
            self.assertEqual(cleanup.respell(said, TERMS), said)

    def test_misspellings_still_swap(self):
        self.assertEqual(cleanup.respell("the wisper flow app", TERMS), "the Wispr Flow app")
        self.assertEqual(cleanup.respell("pull the capitol iq comps", TERMS), "pull the Capital IQ comps")
        self.assertTrue(cleanup.sounds_like("whisper", "Wispr"))
        self.assertFalse(cleanup.sounds_like("is", "IQ"))
        self.assertFalse(cleanup.sounds_like("desks", "Desk"))

    def test_model_made_terms_are_rejected(self):
        """The model may do what respell no longer does: the term guard catches it."""
        self.assertEqual(cleanup.check("we ran the project planning on the new fund",
                                       "We ran the Project Planner on the new fund.", TERMS), "term")
        self.assertEqual(cleanup.check("um so the working capital is tight this quarter",
                                       "So the working Capital IQ tight this quarter.", TERMS), "term")
        self.assertIsNone(cleanup.check("it works like whisper flow and the help desk",
                                        "It works like Wispr Flow and the Help Desk.", TERMS))
        fake = FakeOllama()
        self.addCleanup(fake.close)
        fake.answer("We ran the Project Planner on the new fund last week.")
        c = cleanup.Cleaner(settings(fake.url, model="m", timeout_s=1.0), terms_fn=lambda: TERMS)
        said = "we ran the project planning on the new fund last week"
        self.assertEqual(c.clean(said), said)
        self.assertEqual(c.last["reason"], "guard:term")

    def test_cleaner_pastes_the_phrase_unchanged(self):
        fake = FakeOllama()
        self.addCleanup(fake.close)
        fake.answer("So the working capital is tight this quarter, and we need to watch the costs.")
        c = cleanup.Cleaner(settings(fake.url, model="m", timeout_s=1.0))
        self.assertEqual(c.clean("um so the working capital is tight this quarter and we need to watch the costs"),
                         "So the working capital is tight this quarter, and we need to watch the costs.")
        self.assertEqual(c.last["reason"], "ok")


class TestLearnedPairsAsOrders(unittest.TestCase):
    """F2: only pairs that are safe to apply wherever they are heard reach the model or the guards."""

    def test_order_pair(self):
        for heard, written in (("Kiera", "Kira"), ("blue sky", "BlueSky"), ("iphone", "iPhone"),
                               ("example dot com", "example.com"), ("a lana", "Alana"),
                               ("stellen bosch", "Stellenbosch"), ("q three", "Q3")):
            self.assertTrue(cleanup.order_pair(heard, written), (heard, written))
        for heard, written in (("with", "Alex"), ("mark", "Mark"), ("orca", "ORCA"), ("cache", "cash"),
                               ("yeah", "ja"), ("desk", "deck"), ("ten cent", "Tencent"), ("ten sent", "Tencent"),
                               ("one drive", "OneDrive"), ("first rand", "FirstRand")):
            self.assertFalse(cleanup.order_pair(heard, written), (heard, written))

    def test_unsafe_pairs_are_not_sent(self):
        fake = FakeOllama()
        self.addCleanup(fake.close)
        fake.answer("Send the pack with the notes to Mark about the cache.")
        c = cleanup.Cleaner(settings(fake.url, model="m", timeout_s=1.0),
                            pairs_fn=lambda: [("with", "Alex"), ("mark", "Mark"), ("cache", "cash")])
        c.clean("send the pack with the notes to mark about the cache")
        self.assertNotIn("Spelling for this dictation", fake.requests[-1]["messages"][-1]["content"])

    def test_guard_does_not_allow_unsafe_pair_words(self):
        said = "send the pack with the notes and with the deck to Adam"
        self.assertEqual(cleanup.check(said, "Send the pack Alex the notes and Alex the deck to Adam.", TERMS,
                                       [("with", "Alex")]), "added")

    def test_lexicon_pairs_after_apply(self):
        """How the app should wire it: Lexicon.apply respells the confirmed pairs, pairs_fn=lexicon.pairs
        hands the model the names the speaker has corrected only once."""
        with tempfile.TemporaryDirectory() as tmp:
            lex = learn.Lexicon(Path(tmp) / "lexicon.json")
            lex.add("Kiera", "Kira")                     # a name for a name: waits for a second sighting
            lex.add("stellen bosch", "Stellenbosch")     # confirmed at once
            lex.add("with", "Alex")                       # never used
            text = lex.apply("ask Kiera to drive out to stellen bosch with the deck")
            self.assertEqual(text, "ask Kiera to drive out to Stellenbosch with the deck")
            self.assertEqual(cleanup.relevant_pairs(text, lex.pairs()), [("kiera", "Kira")])


class TestPrefaceGuard(unittest.TestCase):
    """F3 and F7: a preface is caught even when the dictation opens with the same word, and a dictation that
    opens with a filler before "here is" or "sure" is not taken for a preface."""

    def test_preface_on_a_dictation_that_opens_alike(self):
        for said, out in (
                ("here is what I think we should do with the client portfolio before the review on Friday morning",
                 "Here's the cleaned text: Here is what I think we should do with the client portfolio before the "
                 "review on Friday morning."),
                ("I think we should move the meeting to Friday",
                 "I've cleaned it: I think we should move the meeting to Friday."),
                ("the client wants the numbers by friday", "The cleaned dictation: The client wants the numbers by Friday."),
                ("sure I can do the call on friday", "Sure! Here is the cleaned text: Sure, I can do the call on Friday.")):
            self.assertEqual(cleanup.check(said, out, TERMS), "preface", out)

    def test_filler_before_the_opening(self):
        for said, out in (("um here is the plan for tomorrow morning with the client",
                           "Here is the plan for tomorrow morning with the client."),
                          ("uh sure I can do the call on Friday afternoon", "Sure, I can do the call on Friday afternoon."),
                          ("so um absolutely let's do it on Friday", "Absolutely, let's do it on Friday."),
                          ("okay here is the deck for the review", "OK, here's the deck for the review.")):
            self.assertIsNone(cleanup.check(said, out, TERMS), out)

    def test_cleaner_rejects_the_preface(self):
        fake = FakeOllama()
        self.addCleanup(fake.close)
        said = "here is what I think we should do with the client portfolio before the review on Friday morning"
        fake.answer("Here's the cleaned text: Here is what I think we should do with the client portfolio before "
                    "the review on Friday morning.")
        c = cleanup.Cleaner(settings(fake.url, model="m", timeout_s=1.0))
        self.assertEqual(c.clean(said), said)
        self.assertEqual(c.last["reason"], "guard:preface")


class TestAnswerGuards(unittest.TestCase):
    """F4: short answers, echoed answers and answers made of the speaker's own words are rejected."""

    def test_answers(self):
        for said, out in (
                ("send the pack to Adam", "Done."), ("okay send it now", "Sent."),
                ("is the terminal working today", "Yes."), ("is the terminal working today", "It is."),
                ("can you check whether the Bloomberg terminal is working today",
                 "Can you check whether the Bloomberg terminal is working today? The Bloomberg terminal is working today."),
                ("can you check whether the Bloomberg terminal is working today",
                 "Can you check whether the Bloomberg terminal is working today? Yes, it is working."),
                ("what's the best way to structure the blue sky onboarding",
                 "What's the best way to structure the BlueSky onboarding? Structure it in three steps."),
                ("what is the capital of france", "What is the capital of France? Paris.")):
            self.assertIsNotNone(cleanup.check(said, out, TERMS), out)

    def test_cleaner_rejects_a_one_word_answer(self):
        fake = FakeOllama()
        self.addCleanup(fake.close)
        fake.answer("Done.")
        c = cleanup.Cleaner(settings(fake.url, model="m", timeout_s=1.0))
        self.assertEqual(c.clean("send the pack to Adam"), "send the pack to Adam")
        self.assertEqual(c.last["reason"], "guard:added")


class TestFiguresGuard(unittest.TestCase):
    """F5: a number, % or currency the speaker did not say is never pasted."""

    def test_invented_or_changed(self):
        for said, out in (
                ("revenue grew from two point three billion to two point eight billion",
                 "Revenue grew from R2.3 billion to R2.8 billion."),
                ("the fee is one point two five percent", "The fee is 1.75%."),
                ("the fee is one point two five percent", "The fee is one point seven five percent."),
                ("the fund returned well this quarter and costs went down",
                 "The fund returned 12% this quarter, and costs went down 3%."),
                ("send it to Adam on Wednesday morning", "Send it to Adam on Wednesday at 9:00."),
                ("the fee is twenty five thousand rand", "The fee is R25."),
                ("we raised two million dollars", "We raised R2 million."),
                ("the return was four percent", "The return was 4% and 5%.")):
            self.assertEqual(cleanup.check(said, out, TERMS), "figures", out)

    def test_said_figures_pass(self):
        for said, out in (
                ("revenue grew from two point three billion to two point eight billion",
                 "Revenue grew from 2.3 billion to 2.8 billion."),
                ("we need two point three billion rand", "We need R2.3bn."),
                ("the code is one two three four one two three four", "The code is 1234 1234."),
                ("the code is one two three four one two three four", "The code is 12341234."),
                ("we planned it for twenty twenty six", "We planned it for 2026."),
                ("die vergadering is om drie wag om vier", "Die vergadering is om 4."),
                ("let's meet at nine thirty on the sixth", "Let's meet at 9:30 on the 6th."),
                ("let's meet at nine on the sixth", "Let's meet at 9:00 on the 6th."),
                ("we sold two thousand and six units", "We sold 2,006 units."),
                ("the budget is two hundred and fifty thousand rand", "The budget is R250,000."),
                ("costs went up by four and a half percent", "Costs went up by 4.5%."),
                ("die fooi is twee komma vyf persent", "Die fooi is 2,5%."),
                ("send it in Q3", "Send it in Q3.")):
            self.assertIsNone(cleanup.check(said, out, TERMS), out)

    def test_cleaner_rejects_an_invented_currency(self):
        fake = FakeOllama()
        self.addCleanup(fake.close)
        said = "revenue grew from two point three billion to two point eight billion"
        fake.answer("Revenue grew from R2.3 billion to R2.8 billion.")
        c = cleanup.Cleaner(settings(fake.url, model="m", timeout_s=1.0))
        self.assertEqual(c.clean(said), said)
        self.assertEqual(c.last["reason"], "guard:figures")


class TestNumbersNotCollapsed(unittest.TestCase):
    """F6: a repeated run with a number in it is a figure, not a stutter; the model sees all of it."""

    def test_codes_kept(self):
        for said in ("the code is one two three four one two three four", "dial 1 2 3 4 1 2 3 4 now",
                     "send it to Adam on the fifth send it to Adam on the fifth"):
            self.assertEqual(cleanup.collapse_repeats(said), said)

    def test_model_gets_the_whole_code(self):
        fake = FakeOllama()
        self.addCleanup(fake.close)
        fake.answer("The code is 1234 1234.")
        c = cleanup.Cleaner(settings(fake.url, model="m", timeout_s=1.0))
        c.clean("the code is one two three four one two three four")
        self.assertIn("one two three four one two three four", fake.requests[-1]["messages"][-1]["content"])


class TestFalseRejects(unittest.TestCase):
    """F8: contractions and a "scratch that" over an Afrikaans clause are valid clean-ups."""

    def test_contractions(self):
        self.assertIsNone(cleanup.check("I do not think we should and I will not do it",
                                        "I don't think we should, and I won't do it.", TERMS))
        self.assertIsNone(cleanup.check("we cannot do it and they are not ready and I am late",
                                        "We can't do it, and they aren't ready, and I'm late.", TERMS))
        self.assertEqual(cleanup.expand("Don't, won't, can't, I'm, we'll, they’re, I've"),
                         "Do not, will not, can not, I am, we will, they are, I have")

    def test_scratch_that_drops_an_afrikaans_clause(self):
        self.assertIsNone(cleanup.check("ek dink ons moet die client vandag bel scratch that I will call him tomorrow",
                                        "I'll call him tomorrow.", TERMS))


class TestUntestedBehaviour(unittest.TestCase):
    """F9: behaviours the reviewer could break without a test failing."""

    def test_read_timeout_starts_a_warm(self):
        fake = FakeOllama()
        self.addCleanup(fake.close)
        fake.reply = lambda body: (200, {"message": {"content": "x"}, "done_reason": "stop"},
                                   0.0 if body["options"]["num_predict"] == 1 else 1.0)
        c = cleanup.Cleaner(settings(fake.url, model="m", timeout_s=0.3))
        c.clean(SAMPLE)
        self.assertEqual(c.last["reason"], "timeout")
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not any(b["options"]["num_predict"] == 1 for b in fake.requests):
            time.sleep(0.05)
        self.assertTrue(any(b["options"]["num_predict"] == 1 for b in fake.requests))

    def test_connect_wait_is_capped(self):
        seen = []

        def post(url, json=None, timeout=None, proxies=None):
            seen.append(timeout)
            raise requests.ConnectTimeout("no route")
        with mock.patch.object(cleanup.requests, "post", post):
            c = cleanup.Cleaner(settings(model="m", timeout_s=2.5))
            self.assertEqual(c.clean(SAMPLE), SAMPLE)
            self.assertEqual(c.last["reason"], "error")
            cleanup.Cleaner(settings(model="m", timeout_s=0.3)).clean(SAMPLE)
        self.assertEqual(seen, [(cleanup.CONNECT_S, 2.5), (0.3, 0.3)])

    def test_check_gets_the_terms_and_pairs(self):
        fake = FakeOllama()
        self.addCleanup(fake.close)
        fake.answer("Ask about the Zebra Fund Partners deal today, please.")
        c = cleanup.Cleaner(settings(fake.url, model="m", timeout_s=1.0), terms_fn=lambda: ["Zebra Fund Partners"])
        self.assertEqual(c.clean("ask about the zeebra fond partners deal today please"),
                         "Ask about the Zebra Fund Partners deal today, please.")
        fake.answer("Ask Alana to drive out to Stellenbosch for the review.")
        c = cleanup.Cleaner(settings(fake.url, model="m", timeout_s=1.0),
                            pairs_fn=lambda: [("a lana", "Alana"), ("stellen bosch", "Stellenbosch")])
        self.assertEqual(c.clean("ask a lana to drive out to stellen bosch for the review"),
                         "Ask Alana to drive out to Stellenbosch for the review.")
        self.assertEqual(c.last["reason"], "ok")

    def test_afrikaans_drop_each_half(self):
        # Most of the distinct Afrikaans words gone, though the share stays up.
        self.assertEqual(cleanup.check("ek dink ons moet die client vandag bel",
                                       "Ek dink, ek dink we call the client today.", TERMS), "translation")
        # Every distinct Afrikaans word kept once, but English took over the rest.
        self.assertEqual(cleanup.check(
            "ek dink ons moet die fees verlaag en ek dink ons moet die client vandag bel",
            "Ek dink ons moet die fees verlaag en we must call the client today, so we lower the fees and call "
            "him today", TERMS), "translation")


# ---------------------------------------------------------------- review fixes 2 (31 Oct 2026)

def without_proxy_settings() -> dict:
    """os.environ without any *_PROXY variable, for mock.patch.dict(..., clear=True)."""
    return {k: v for k, v in os.environ.items() if not k.lower().endswith("_proxy")}


class TestFigureOrder(unittest.TestCase):
    """privacy-failure F1: the speaker's figures come back in the order the speaker said them, each used once."""

    def test_swapped_repeated_or_changed(self):
        for said, out in (
                ("we need three people for two weeks", "We need 2 people for 3 weeks."),
                ("revenue was twelve million rand and costs were twenty million rand",
                 "Revenue was R20 million and costs were R12 million."),
                ("margin went from twelve percent to twenty percent", "Margin went from 20% to 12%."),
                ("we need three people", "We need 3 people for 3 weeks."),
                ("Book the room for twelve people, I mean fifteen people", "Book the room for 12 people."),
                ("die vergadering is om drie wag om vier", "Die vergadering is om 3."),
                ("we close the books on the fifth of the month", "We close the books on the sixth of the month."),
                ("the fund lost billions this year", "The fund lost millions this year."),
                ("ons het vyf en twintig mense nodig vir die projek", "Ons het vyfendertig mense nodig vir die projek."),
                ("we need three people for two weeks", "We need people for 32 weeks."),
                ("the code is one two three", "The code is one two two three."),
                ("ons is dit eens oor die plan", "Ons is dit een oor die plan."),
                # A joined figure may not reuse a figure already taken (take()'s start bound).
                ("we planned it for twenty twenty six", "We planned 20 for 2026."),
                ("the pin is one two three four", "The pin is 1, 1234.")):
            self.assertEqual(cleanup.check(said, out, TERMS), "figures", out)

    def test_his_order_passes(self):
        for said, out in (
                ("costs rose twelve percent and revenue twenty percent", "Costs rose 12% and revenue 20%."),
                ("we need three people for two weeks", "We need 3 people for 2 weeks."),
                ("ten no wait eleven people for the dinner", "11 people for the dinner."),
                ("Book the room for twelve people, I mean fifteen people, on Thursday.",
                 "Book the room for 15 people on Thursday."),
                ("we need three people actually three people and a projector", "We need 3 people and a projector."),
                ("ons het vyfentwintig mense nodig", "Ons het 25 mense nodig."),
                ("ons het tweehonderd mense en drieënveertig stoele", "Ons het 200 mense en 43 stoele."),
                ("firstly we close the books on the fifth", "First, we close the books on the 5th."),
                ("the fund lost billions this year", "The fund lost billions this year.")):
            self.assertIsNone(cleanup.check(said, out, TERMS), out)
        # Spoken numbers in the output may read as one of the speaker's figures (the added guard counts the words).
        self.assertTrue(cleanup.figures_ok("we planned it for 2026", "We planned it for twenty twenty six."))
        self.assertFalse(cleanup.figures_ok("we planned it for 2027", "We planned it for twenty twenty six."))

    def test_compound_repeats_are_figures(self):
        """learn.number_words reads Afrikaans compounds, so a repeat holding one is a figure, left to the model."""
        said = "ons het vyfentwintig mense ons het vyfentwintig mense nodig"
        self.assertEqual(cleanup.collapse_repeats(said), said)


class TestNegationGuard(unittest.TestCase):
    """privacy-failure F3: a negation lost or added reverses the meaning; a stutter or a correction does not."""

    def test_reversals(self):
        for said, out in (
                ("we should not ship the release to the client today", "We should ship the release to the client today."),
                ("I don't think we should sign the contract yet", "I think we should sign the contract yet."),
                ("we can't send the pack to the client before Friday", "We can send the pack to the client before Friday."),
                ("we never agreed to the fee increase", "We agreed to the fee increase."),
                ("ek gaan nie die kontrak vandag teken nie", "Ek gaan die kontrak vandag teken."),
                ("ek is nie seker oor die fee nie", "Ek is seker oor die fee."),
                ("there is no reason to wait for the client", "There is reason to wait for the client."),
                ("ons het geen tyd vir die vergadering nie", "Ons het tyd vir die vergadering."),
                ("we won't sign the contract this week", "We will sign the contract this week."),
                ("um we should uh not ship it today", "We should ship it today."),
                ("the fee is fifteen percent not fifty percent", "The fee is 50%."),
                ("we should ship the release to the client today", "We should not ship the release to the client today."),
                ("we should not actually ship it this week", "We should actually ship it this week."),
                ("we should not ship the release actually", "We should ship the release, actually."),
                ("sorry we should not ship the release today", "Sorry, we should ship the release today."),
                ("we will not sign the contract with the client this week because the numbers are still open "
                 "actually we can talk on Monday", "We will sign the contract with the client this week because the "
                 "numbers are still open. Actually, we can talk on Monday.")):
            self.assertEqual(cleanup.check(said, out, TERMS), "negation", out)

    def test_each_negation_is_judged_on_its_own(self):
        """A correction marker takes away only a negation just before it, not every negation the speaker said."""
        said = "the client is not happy with the fee actually he is fine with the fee but not the timeline"
        self.assertEqual(cleanup.check(said, "The client is happy with the fee.", TERMS), "negation")
        self.assertIsNone(cleanup.check(said, "The client is fine with the fee, but not the timeline.", TERMS))

    def test_stutters_and_corrections_pass(self):
        for said, out in (
                ("the numbers are not not final", "The numbers are not final."),
                ("we can not we can not do it", "We can't do it."),
                ("no we should not ship it", "We should not ship it."),
                ("no wait we should send it on Friday", "We should send it on Friday."),
                ("we should not ship it today sorry we should ship it tomorrow", "We should ship it tomorrow."),
                ("ek gaan nie vandag teken nie nee ek bedoel ek gaan môre teken", "Ek gaan môre teken."),
                ("ek is nie seker nie", "Ek is nie seker nie."),
                ("ek is nie seker", "Ek is nie seker nie."),
                ("ek is nie seker oor die fee nie", "Ek is nie seker oor die fee."),     # a pair may fall to one
                ("there is no time to waste on this", "There's no time to waste on this."),
                ("No, Friday works better for the client", "Friday works better for the client."),
                ("I do not think so", "I don't think so.")):
            self.assertIsNone(cleanup.check(said, out, TERMS), out)


class TestAfrikaansTwins(unittest.TestCase):
    """privacy-failure F4: one or two Afrikaans words swapped for their English twins, or back, are a
    translation, however short the code-switch."""

    def test_swaps(self):
        for said, out in (
                ("die meeting was goed but I think we need to send the minutes to everyone by Friday",
                 "The meeting was good, but I think we need to send the minutes to everyone by Friday."),
                ("send the minutes en die agenda to everyone by Friday",
                 "Send the minutes and the agenda to everyone by Friday."),
                ("dis fine we can send the pack tomorrow", "This is fine, we can send the pack tomorrow."),
                ("dis fine we can send the pack tomorrow", "It's fine, we can send the pack tomorrow."),
                ("die client wants the report by Friday", "The client wants the report by Friday."),
                ("send it to Adam maar he must check the numbers first", "Send it to Adam, but he must check the numbers first."),
                ("send the deck to the client vandag please", "Send the deck to the client today, please."),
                ("send die minutes to die client by Friday", "Send die minutes to the client by Friday."),
                ("send die minutes and the agenda to everyone by Friday",
                 "Send die minutes and die agenda to everyone by Friday.")):
            self.assertEqual(cleanup.check(said, out, TERMS), "translation", out)

    def test_code_switch_kept(self):
        for said, out in (
                ("die meeting was goed but I think we need to send the minutes to everyone by Friday",
                 "Die meeting was goed, but I think we need to send the minutes to everyone by Friday."),
                ("dis fine we can send the pack tomorrow", "Dis fine, we can send the pack tomorrow."),
                ("send the minutes en die agenda to everyone by Friday",
                 "Send the minutes en die agenda to everyone by Friday."),
                ("send it to die nee ek bedoel the client", "Send it to the client."),
                ("send die die report to the client", "Send die report to the client."),
                ("ja okay send the pack to adam today", "OK, send the pack to Adam today."),
                ("send the pack to adam sommer today okay", "Send the pack to Adam today, OK?"),
                ("I met the client van Wyk on Friday", "I met the client Van Wyk on Friday.")):
            self.assertIsNone(cleanup.check(said, out, TERMS), out)


class TestQuestionsStayQuestions(unittest.TestCase):
    """privacy-failure F5: an answer word not said, the speaker's question turned into a statement, or a sentence of
    the speaker's words that earlier sentences used up is an answer, whatever the allowance for new words."""

    def test_answers(self):
        sqlite = "I want to set up the database for the workbench this week should we use postgres or sqlite for this"
        for said, out in (
                ("is the terminal working today", "Yes, the terminal is working today."),
                ("should we send the pack to Alana today", "Yes, we should send the pack to Alana today."),
                ("is the Bloomberg terminal working today", "Yes, the Bloomberg terminal is working today."),
                ("is the terminal working today", "Yes, it is working."),
                ("Is the terminal working today?", "Yes, the terminal is working today."),
                ("Is the terminal working today?", "The terminal is working today."),
                ("Kan jy vir my sê of die vergadering môre is?", "Die vergadering is môre."),
                ("What is the plan for the client meeting?", "The plan for the client meeting."),
                ("So can you check the numbers before the call?", "You can check the numbers before the call."),
                ("send the pack to Adam before Friday", "OK, I'll send the pack to Adam before Friday."),
                (sqlite, "I want to set up the database for the workbench this week. Should we use Postgres or "
                         "SQLite for this? Use SQLite."),
                (sqlite, "Use SQLite. I want to set up the database for the workbench this week. Should we use "
                         "Postgres or SQLite for this?")):
            self.assertEqual(cleanup.check(said, out, TERMS), "added", out)

    def test_the_speakers_questions_and_statements_pass(self):
        for said, out in (
                ("Is the terminal working today?", "Is the terminal working today?"),
                ("um is the uh terminal working today", "Is the terminal working today?"),
                ("so what's the best way to structure the blue sky onboarding",
                 "So what's the best way to structure the BlueSky onboarding?"),
                ("kan jy vir my sê hoe laat die vergadering môre is", "Kan jy vir my sê hoe laat die vergadering môre is?"),
                ("yes so the terminal is working today", "Yes, so the terminal is working today."),
                ("okay send the pack to Adam", "OK, send the pack to Adam."),
                ("the report is done the report is good", "The report is done. The report is good."),
                ("do the review on friday with the team", "Do the review on Friday with the team."),
                ("what I think is that we should wait for the numbers",
                 "What I think is that we should wait for the numbers."),
                ("when you get a chance send me the deck", "When you get a chance, send me the deck."),
                ("I need the following blue sky and capital iq", "I need the following:\n- BlueSky\n- Capital IQ")):
            self.assertIsNone(cleanup.check(said, out, TERMS), out)
        self.assertIsNone(cleanup.check("the agenda is kiera and the budget", "The agenda:\n- Kira\n- The budget",
                                        TERMS, [("Kiera", "Kira")]))

    def test_cleaner_pastes_the_speakers_text(self):
        """Each new guard ends in Whisper's text, with its reason."""
        fake = FakeOllama()
        self.addCleanup(fake.close)
        c = cleanup.Cleaner(settings(fake.url, model="m", timeout_s=1.0))
        for said, reply, reason in (
                ("is the terminal working today", "Yes, the terminal is working today.", "guard:added"),
                ("we should not ship the release today", "We should ship the release today.", "guard:negation"),
                ("send the minutes en die agenda to everyone", "Send the minutes and the agenda to everyone.",
                 "guard:translation"),
                ("we need three people for two weeks", "We need 2 people for 3 weeks.", "guard:figures")):
            fake.answer(reply)
            with self.subTest(reason):
                self.assertEqual(c.clean(said), said)
                self.assertEqual(c.last["reason"], reason)


class TestNoProxy(unittest.TestCase):
    """privacy-failure F6: the dictation goes only to the local server, never to a proxy set in HTTP_PROXY,
    ALL_PROXY or the Windows settings."""

    def test_proxy_never_sees_the_dictation(self):
        fake, proxy = FakeOllama(), FakeOllama()
        self.addCleanup(fake.close)
        self.addCleanup(proxy.close)
        fake.answer("So I think we should move the meeting to Friday.")
        system = mock.patch.object(requests.utils, "getproxies", lambda: {"http": proxy.url, "https": proxy.url})
        for name, env, patcher in (("HTTP_PROXY", {"HTTP_PROXY": proxy.url}, None),
                                   ("ALL_PROXY", {"ALL_PROXY": proxy.url}, None),
                                   ("Windows", {}, system)):
            with self.subTest(name), mock.patch.dict(os.environ, {**without_proxy_settings(), **env}, clear=True):
                if patcher:
                    patcher.start()
                try:
                    c = cleanup.Cleaner(settings(fake.url, model="m", timeout_s=1.0))
                    self.assertEqual(c.clean(SAMPLE), "So I think we should move the meeting to Friday.")
                    c.warm(wait=True)
                finally:
                    if patcher:
                        patcher.stop()
                self.assertEqual(proxy.requests, [])
        self.assertEqual(len(fake.requests), 6)

    def test_the_setting_is_never_changed(self):
        """requests adds the environment's other proxies to the dict it is given: each call gets a copy."""
        saved = dict(cleanup.NO_PROXY)
        self.addCleanup(lambda: (cleanup.NO_PROXY.clear(), cleanup.NO_PROXY.update(saved)))
        fake, proxy = FakeOllama(), FakeOllama()
        self.addCleanup(fake.close)
        self.addCleanup(proxy.close)
        fake.answer("So I think we should move the meeting to Friday.")
        env = {**without_proxy_settings(), "FTP_PROXY": proxy.url, "NO_PROXY": "localhost"}
        with mock.patch.dict(os.environ, env, clear=True):
            c = cleanup.Cleaner(settings(fake.url, model="m", timeout_s=1.0))
            self.assertEqual(c.clean(SAMPLE), "So I think we should move the meeting to Friday.")
            c.warm(wait=True)
        self.assertEqual(cleanup.NO_PROXY, {"http": None, "https": None, "all": None})

    def test_think_retry_skips_the_proxy_too(self):
        fake, proxy = FakeOllama(), FakeOllama()
        self.addCleanup(fake.close)
        self.addCleanup(proxy.close)
        fake.reply = lambda body: ((400, {"error": '"m" does not support thinking'}, 0) if "think" in body else
                                   (200, {"message": {"content": "So I think we should move the meeting to Friday."},
                                          "done_reason": "stop"}, 0))
        proxy.reply = fake.reply
        with mock.patch.dict(os.environ, {**without_proxy_settings(), "HTTP_PROXY": proxy.url}, clear=True):
            c = cleanup.Cleaner(settings(fake.url, model="m", timeout_s=1.0))
            self.assertEqual(c.clean(SAMPLE), "So I think we should move the meeting to Friday.")
        self.assertEqual(["think" in b for b in fake.requests], [True, False])
        self.assertEqual(proxy.requests, [])


# ---------------------------------------------------------------- review fixes 3 (31 Oct 2026)

class TestNieAndOtherNegators(unittest.TestCase):
    """F3 remainder: each "nie" pair the speaker said is kept, none is added, and without/sonder and neither/nor
    negate too."""

    def test_a_pair_lost_or_added(self):
        for said, out in (
                ("ek kan nie Maandag kom nie maar ek wil nie Dinsdag gaan nie",
                 "Ek kan nie Maandag kom nie, maar ek wil Dinsdag gaan."),
                ("ons stuur dit nie vandag nie ons stuur dit nie môre nie",
                 "Ons stuur dit nie vandag nie, ons stuur dit môre."),
                ("ek kan nie kom nie maar ek wil gaan na die meeting",
                 "Ek kan nie kom nie, maar ek wil nie gaan na die meeting."),
                ("ek kan kom maar ek wil nie gaan nie", "Ek kan nie kom, maar ek wil nie gaan nie.")):
            self.assertEqual(cleanup.check(said, out, TERMS), "negation", out)

    def test_without_and_neither_nor(self):
        for said, out in (
                ("we can ship it without the sign-off", "We can ship it with the sign-off."),
                ("we can ship it with the sign-off", "We can ship it without the sign-off."),
                ("ons kan nie sonder die numbers begin nie", "Ons kan nie die numbers begin nie."),
                ("we can neither ship it nor sign it this week", "We can ship it and sign it this week."),
                ("we can neither ship it nor sign it this week", "We can neither ship it and sign it this week.")):
            self.assertEqual(cleanup.check(said, out, TERMS), "negation", out)

    def test_his_negations_pass(self):
        two_pairs = "ek kan nie Maandag kom nie maar ek wil nie Dinsdag gaan nie"
        for said, out in (
                (two_pairs, "Ek kan nie Maandag kom nie, maar ek wil nie Dinsdag gaan nie."),
                (two_pairs, "Ek kan nie Maandag kom nie, maar ek wil nie Dinsdag gaan."),     # one pair falls to one
                ("ek is nie seker", "Ek is nie seker nie."),                                  # an odd one closed
                ("we can ship it without the sign-off", "We can ship it without the sign-off."),
                ("ons kan nie sonder die numbers begin nie", "Ons kan nie sonder die numbers begin nie."),
                ("we can neither ship it nor sign it this week", "We can neither ship it nor sign it this week.")):
            self.assertIsNone(cleanup.check(said, out, TERMS), out)


class TestAfrikaansWordsOutsideTheTwins(unittest.TestCase):
    """F4 remainder: a spoken Afrikaans word gone while an unsaid word came in is a translation,
    whether or not TWINS lists the word."""

    def test_swaps(self):
        for said, out in (
                ("the meeting was lekker", "The meeting was nice."),
                ("send the pack gou to Adam", "Send the pack quickly to Adam."),
                ("we must sommer send it to Adam", "We must just send it to Adam."),
                ("he is nog busy with the pack", "He is still busy with the pack."),
                ("can you kry the numbers from Adam", "Can you get the numbers from Adam?"),
                ("we sent the pack toe he can check it", "We sent the pack, then he can check it."),
                ("send the pack to jou manager today", "Send the pack to your manager today."),
                ("send the epos to Adam before Friday", "Send the email to Adam before Friday.")):
            self.assertEqual(cleanup.check(said, out, TERMS), "translation", out)

    def test_kept_or_dropped_passes(self):
        for said, out in (
                ("the meeting was lekker", "The meeting was lekker."),
                ("send the pack gou to Adam", "Send the pack gou to Adam."),
                ("ja okay send the pack to adam today", "OK, send the pack to Adam today."),
                ("send the pack to adam sommer today okay", "Send the pack to Adam today, OK?"),
                ("send it to die nee ek bedoel the client", "Send it to the client."),
                ("ek dink ons moet die client vandag bel scratch that I will call him tomorrow",
                 "I'll call him tomorrow.")):
            self.assertIsNone(cleanup.check(said, out, TERMS), out)


class TestStuttersWithANegation(unittest.TestCase):
    """Regression from F3: a say-it-twice repeat of any length, or one with a filler between the copies, is
    one negation, as the model writes it once."""

    def test_long_repeat_with_a_negation(self):
        """collapse_repeats hands the model one copy of a 4+ word repeat; the guard reads the dictation as one
        copy too."""
        fake = FakeOllama()
        self.addCleanup(fake.close)
        fake.answer("I don't think we should sign the contract.")
        c = cleanup.Cleaner(settings(fake.url, model="m", timeout_s=2.0), terms_fn=lambda: TERMS)
        said = "I don't think we should, I don't think we should sign the contract."
        self.assertEqual(c.clean(said), "I don't think we should sign the contract.")
        self.assertEqual(fake.requests[-1]["messages"][-1]["content"].count("don't"), 1)     # the model got one copy
        self.assertEqual(c.last["reason"], "ok")

    def test_stutters_pass(self):
        for said, out in (
                ("we can't um we can't send it today", "We can't send it today."),
                ("we should not uh not ship it", "We should not ship it."),
                ("there is no, there is no reason to wait", "There is no reason to wait."),
                ("I think there is no, I think there is no reason to wait", "I think there is no reason to wait."),
                ("we will not sign it this week we will not sign it this week", "We will not sign it this week.")):
            self.assertIsNone(cleanup.check(said, out, TERMS), out)

    def test_reversals_still_caught(self):
        for said, out in (
                ("we can't um we can't send it today", "We can send it today."),
                ("we should not uh not ship it", "We should ship it."),
                ("I don't think we should, I don't think we should sign the contract.",
                 "I think we should sign the contract."),
                ("there is no, there is no reason to wait", "There is reason to wait.")):
            self.assertEqual(cleanup.check(said, out, TERMS), "negation", out)

    def test_long_text_is_fast(self):
        long = " ".join(f"we did not send word{i % 97} and the other{i % 89}" for i in range(60))
        t0 = time.perf_counter()
        self.assertTrue(cleanup.negation_ok(long, long))
        self.assertLess(time.perf_counter() - t0, 0.5)


class TestListRequests(unittest.TestCase):
    """Regression from F5: a request asked as a question keeps its question in the prompt's list format, a
    lead-in ending ':' over the list."""

    def test_request_with_a_list(self):
        for said, out in (
                ("can you do the following three things first check the pack second send the deck and third "
                 "book the call", "Can you do the following 3 things:\n1. Check the pack.\n2. Send the deck.\n"
                                  "3. Book the call."),
                ("Can you check the following for me, the pack, the deck and the model?",
                 "Can you check the following for me:\n- The pack\n- The deck\n- The model")):
            self.assertIsNone(cleanup.check(said, out, TERMS), out)

    def test_a_statement_is_still_caught(self):
        for said, out in (("Can you check the following for me, the pack, the deck and the model?",
                           "Check the following for me: the pack, the deck and the model."),
                          ("Can you check the pack? Can you send the deck?",
                           "Can you check the following:\n- The pack\n- The deck")):
            self.assertEqual(cleanup.check(said, out, TERMS), "added", out)


class TestTermLines(unittest.TestCase):
    """Regression from F5: a line or sentence that is only a name from the terms is worth one use, since the speaker
    may have said it as a sound-alike; a second use is still an answer."""

    def test_a_line_that_is_only_a_respelt_term(self):
        self.assertIsNone(cleanup.check("send it to the following people sam alana and kiera",
                                        "Send it to the following people:\n- Sam\n- Alana\n- Kira", TERMS))

    def test_a_repeated_term_is_an_answer(self):
        self.assertEqual(cleanup.check("Is Kira in?", "Is Kira in? Kira.", TERMS), "added")
        self.assertEqual(cleanup.check("is kira in today", "Is Kira in today? Kira.", TERMS), "added")


class TestNoAsAnInterjection(unittest.TestCase):
    """Regression from F3: on a dictation with no punctuation, the model's comma after "no" does not turn a
    determiner into an interjection, and "no thanks" is an interjection."""

    def test_unpunctuated_no_passes(self):
        for said, out in (("no Adam said he would come on Friday", "No, Adam said he would come on Friday."),
                          ("no Friday works for me", "No, Friday works for me."),
                          ("the answer is no thanks", "The answer is no, thanks."),
                          ("he was a no-show at the meeting", "He was a no-show at the meeting.")):
            self.assertIsNone(cleanup.check(said, out, TERMS), out)

    def test_no_thanks(self):
        for said in ("The answer is no thanks.", "No thanks, I will call Adam myself."):
            self.assertIsNone(cleanup.check(said, said.replace("o thanks", "o, thanks"), TERMS), said)

    def test_determiner_still_caught(self):
        for said, out in (("there is no reason to wait", "There is reason to wait."),
                          ("there is no Bloomberg data for it", "There is Bloomberg data for it."),
                          ("There is no reason to wait.", "There is reason to wait."),
                          ("no Adam said he would come on Friday", "Adam said he would come on Friday.")):
            self.assertEqual(cleanup.check(said, out, TERMS), "negation", out)


class TestQuestionsByQuestionMarks(unittest.TestCase):
    """Ruling: a question is one the speaker ended with '?'. The opening-word test is gone, so an unpunctuated
    statement that opens like a question passes; an answer word the speaker did not say is still refused."""

    def test_unpunctuated_dictations_pass(self):
        for said, out in (("should you need anything let me know", "Should you need anything, let me know."),
                          ("what's interesting is the margin this quarter",
                           "What's interesting is the margin this quarter."),
                          ("which is why we should ship it", "Which is why we should ship it."),
                          ("is the terminal working today", "The terminal is working today.")):
            self.assertIsNone(cleanup.check(said, out, TERMS), out)

    def test_answer_words_still_refused(self):
        for said, out in (("is the terminal working today", "Yes, the terminal is working today."),
                          ("should we send the pack to Alana today", "Yeah, we should send the pack to Alana today."),
                          ("can you check the numbers before the call", "Yep, I can check the numbers before the call.")):
            self.assertEqual(cleanup.check(said, out, TERMS), "added", out)


class TestNumberNamesAreNotOrders(unittest.TestCase):
    """learn.Lexicon may keep 'ten cent' -> 'Tencent' as a hint, but the model is never told to write the name
    wherever it hears the words: 'it costs ten cent' is a figure."""

    def test_not_sent_to_the_model(self):
        pairs = [("ten cent", "Tencent"), ("one drive", "OneDrive"), ("Kiera", "Kira")]
        self.assertEqual(cleanup.relevant_pairs("ask kiera to buy ten cent shares and save it to one drive", pairs),
                         [("Kiera", "Kira")])
        fake = FakeOllama()
        self.addCleanup(fake.close)
        fake.answer("The coffee costs ten cent more today.")
        c = cleanup.Cleaner(settings(fake.url, model="m", timeout_s=1.0), pairs_fn=lambda: pairs)
        c.clean("the coffee costs ten cent more today")
        self.assertNotIn("Tencent", fake.requests[-1]["messages"][-1]["content"])

    def test_figures_as_digits_still_apply(self):
        self.assertTrue(cleanup.order_pair("q three", "Q3"))
        self.assertTrue(cleanup.order_pair("seven eleven", "7-Eleven"))


# ---------------------------------------------------------------- review fixes 4 (31 Oct 2026)

class TestMoenieOpensANiePair(unittest.TestCase):
    """F3 remainder: "moenie" (moet nie) opens a "nie" pair, so it counts with the "nie"s."""

    def test_a_moenie_pair_dropped(self):
        said = "moenie dit stuur nie en moenie vir Adam bel nie"
        self.assertEqual(cleanup.check(said, "Moenie dit stuur nie, en bel vir Adam.", TERMS), "negation")
        self.assertIsNone(cleanup.check(said, "Moenie dit stuur nie, en moenie vir Adam bel nie.", TERMS))
        self.assertIsNone(cleanup.check("moenie dit vandag stuur nie", "Moenie dit vandag stuur.", TERMS))  # one pair


class TestHierdieAndDaardie(unittest.TestCase):
    """F4 remainder: their English is a FREE word, which the general rule never sees, so TWINS lists them."""

    def test_hierdie_and_daardie(self):
        for said, out in (("send hierdie deck to Adam today", "Send this deck to Adam today."),
                          ("send daardie deck to Adam today", "Send that deck to Adam today."),
                          ("ek wil hierdie deck vandag stuur", "Ek wil this deck vandag stuur.")):
            with self.subTest(said):
                self.assertEqual(cleanup.check(said, out, TERMS), "translation")
        self.assertIsNone(cleanup.check("send hierdie deck to Adam today", "Send hierdie deck to Adam today.", TERMS))

    def test_a_stuttered_twin(self):
        """A stutter written once is no word gone, so a FREE "that" said elsewhere is no swap; the stutter
        swapped whole still is."""
        for said, out in (("i think daardie daardie deck is fine", "I think that daardie deck is fine."),
                          ("we know daardie daardie numbers are right", "We know that daardie numbers are right."),
                          ("i think daardie um daardie deck is fine", "I think that daardie deck is fine.")):
            with self.subTest(said):
                self.assertIsNone(cleanup.check(said, out, TERMS))
        self.assertEqual(cleanup.check("i think daardie daardie deck is fine", "I think that deck is fine.", TERMS),
                         "translation")


class TestAnswersMadeOfAName(unittest.TestCase):
    """Regression from F5: a word of a name from the terms is worth one use only when the speaker said a sound-alike of
    it, so a one-name answer the speaker did not say is refused; '- Kira' for 'kiera' still passes."""

    def test_an_answer_that_is_only_a_term(self):
        for said, out in (("Which terminal should we use for the comps?",
                           "Which terminal should we use for the comps? Bloomberg."),
                          ("who should run the memory system", "Who should run the memory system? Kira."),
                          ("which model should we use for the build", "Which model should we use for the build? Opus."),
                          ("which terminal has the best bond data", "Bloomberg. Which terminal has the best bond data?")):
            with self.subTest(out):
                self.assertEqual(cleanup.check(said, out, TERMS), "added")
        self.assertIsNone(cleanup.check("send it to the following people sam alana and kiera",
                                        "Send it to the following people:\n- Sam\n- Alana\n- Kira", TERMS))


class TestAListKeepsItsOwnQuestion(unittest.TestCase):
    """Regression from F5: a ':' lead-in over a list counts as a kept question only when it is made of one of their
    questions, so a question answered beside a list is refused."""

    def test_a_list_does_not_stand_in_for_another_question(self):
        said = "Should we use Postgres or SQLite? I need the following, the tables, the indexes and the views."
        self.assertEqual(cleanup.check(said, "We should use SQLite. I need the following:\n- The tables\n"
                                             "- The indexes\n- The views", TERMS), "added")
        self.assertIsNone(cleanup.check(said, "Should we use Postgres or SQLite? I need the following:\n"
                                              "- The tables\n- The indexes\n- The views", TERMS))
        self.assertIsNone(cleanup.check("Can you check the following for me, the pack, the deck and the model?",
                                        "Can you check the following for me:\n- The pack\n- The deck\n- The model",
                                        TERMS))

    def test_a_lead_in_with_too_few_of_the_speakers_words(self):
        """A one-word lead-in the speaker did not say, or one of FREE words only, is not the speaker's question."""
        for said, out in (("Which database should we use? Here are the requirements, speed, cost and support.",
                           "We should use Postgres. Requirements:\n- Speed\n- Cost\n- Support"),
                          ("Can you check these for me? The pack, the deck and the model.",
                           "It is:\n- The pack\n- The deck\n- The model")):
            with self.subTest(out):
                self.assertEqual(cleanup.check(said, out, TERMS), "added")

    def test_a_respelt_term_in_a_lead_in(self):
        """The lead-in is matched against the speaker's question respelt, so a term the model respelt there as told is
        not a new word; a name the speaker did not say still is."""
        for said, out in (("Can you pull these from morning star, the prices and the ratings?",
                           "Can you pull the following from Morningstar:\n- The prices\n- The ratings"),
                          ("Can you send these to whisper flow, the pack and the deck?",
                           "Can you send the following to Wispr Flow:\n- The pack\n- The deck"),
                          ("Can you pull these from blue sky, the prices and the ratings?",
                           "Can you pull the following from BlueSky:\n- The prices\n- The ratings"),
                          ("Can you ask kiera to check these, the pack and the deck?",
                           "Can you ask Kira to check the following:\n- The pack\n- The deck")):
            with self.subTest(said):
                self.assertIsNone(cleanup.check(said, out, TERMS))
        self.assertEqual(cleanup.check("Can you check these for me, the pack and the deck?",
                                       "Can you check the following with Bloomberg:\n- The pack\n- The deck", TERMS),
                         "added")


class TestAfrikaansWordsAsTheModelKeepsThem(unittest.TestCase):
    """Regression from F4: the general rule counts the speaker's Afrikaans words as the model is told to keep them, so a
    stutter's second copy or a word a correction removed is not a word lost, English homographs included."""

    def test_a_stuttered_or_corrected_afrikaans_word_with_one_new_word(self):
        for said, out in (("dan um dan said we're gonna need the deck", "Dan said we're going to need the deck."),
                          ("send it to dan no wait to adam we're gonna need it",
                           "Send it to Adam, we're going to need it."),
                          ("ek dink ons ons is gonna miss the deadline", "Ek dink ons is going to miss the deadline."),
                          ("ons moet die die cache flows check before friday",
                           "Ons moet die cash flows check before Friday.")):
            with self.subTest(said):
                self.assertIsNone(cleanup.check(said, out, TERMS))
        self.assertEqual(cleanup.check("the meeting was lekker", "The meeting was nice.", TERMS), "translation")

    def test_markers_and_plain_words(self):
        """"ek bedoel" is a marker the model drops, not their Afrikaans; "actually" straight after a word is a plain
        word, so the word before it is still the speaker's."""
        self.assertIsNone(cleanup.check("send it to die client ek bedoel to adam we're gonna need it",
                                        "Send it to Adam, we're going to need it.", TERMS))
        self.assertEqual(cleanup.check("the meeting was lekker actually", "The meeting was nice, actually.", TERMS),
                         "translation")


class TestNoThanksAsADeterminer(unittest.TestCase):
    """Regression from F3: a "no" straight before "thanks" or "thank" always counts, on both texts, so dropping it
    is refused wherever it stands; "got no thanks for the work" negates, and "No, thanks" is their "no thanks"."""

    def test_no_thanks_as_a_determiner(self):
        for said in ("he got no thanks for all the work", "He got no thanks for all the work."):
            with self.subTest(said):
                self.assertEqual(cleanup.check(said, "He got thanks for all the work.", TERMS), "negation")
        for said, out in (("the answer is no thanks", "The answer is no, thanks."),
                          ("No thanks, I will call Adam myself.", "No, thanks, I will call Adam myself."),
                          ("no thank you I will manage", "No, thank you. I will manage."),
                          ("No thanks, Adam will drive.", "No, thanks, Adam will drive.")):
            with self.subTest(said):
                self.assertIsNone(cleanup.check(said, out, TERMS))

    def test_no_before_thanks_dropped(self):
        """At the end of the text, before punctuation and before a word in _NO_AFTER the "no" still counts, and
        so does the "no" of a decline."""
        for said, out in (("for all that work he got no thanks", "For all that work, he got thanks."),
                          ("For all that work he got no thanks.", "For all that work, he got thanks."),
                          ("He got no thanks, not even a call.", "He got thanks, not even a call."),
                          ("He got no thanks, which hurt.", "He got thanks, which hurt."),
                          ("he got no thanks and no pay for it", "He got thanks and no pay for it."),
                          ("he got no thanks the whole year", "He got thanks the whole year."),
                          ("No thanks, I will call Adam myself.", "Thanks, I will call Adam myself.")):
            with self.subTest(said):
                self.assertEqual(cleanup.check(said, out, TERMS), "negation")
        for said, out in (("The answer is no thanks.", "The answer is no, thanks."),
                          ("No thank you, I will manage.", "No, thank you, I will manage."),
                          ("no thanks i am fine", "No, thanks, I am fine."),
                          ("I said no thanks, and left.", "I said no, thanks, and left.")):
            with self.subTest(said):
                self.assertIsNone(cleanup.check(said, out, TERMS))


class TestRound2Gaps(unittest.TestCase):
    """Pieces of the round-2 fix no other test pins: each fails on its own revert."""

    def test_neither_on_its_own(self):
        """No other test drops 'neither' while 'nor' stays."""
        self.assertEqual(cleanup.check("neither option works for us", "Either option works for us.", TERMS),
                         "negation")

    def test_no_thank_you(self):
        self.assertIsNone(cleanup.check("No thank you, I will manage.", "No, thank you, I will manage.", TERMS))

    def test_other_list_marks(self):
        """Only '- ' and '1. ' lists are tested elsewhere."""
        said = "Can you check the following for me, the pack, the deck and the model?"
        for out in ("Can you check the following for me:\n* The pack\n* The deck\n* The model",
                    "Can you check the following for me:\n1) The pack\n2) The deck\n3) The model"):
            with self.subTest(out):
                self.assertIsNone(cleanup.check(said, out, TERMS))

    def test_one_of_two_uses_swapped(self):
        """Every other general-rule test drops a word said once."""
        self.assertEqual(cleanup.check("die meeting is lekker en die agenda is lekker",
                                       "Die meeting is lekker en die agenda is nice.", TERMS), "translation")


class TestNorAfterANegation(unittest.TestCase):
    """A "nor" after another negator in its sentence is part of that negation: it may trade places with an "or"
    ('did not send the deck nor the pack' -> 'did not send the deck or the pack'). A "nor" that opens a clause
    ("nor does Adam") or a sentence ("Nor Sarah."), or has no negator before it, negates on its own; without a new
    "or" a "nor" still counts, and an "or" written as a "nor" is a negation added."""

    def test_nor_written_as_or_passes(self):
        for said, out in (("we did not send the deck nor the pack", "We did not send the deck or the pack."),
                          ("there is no data nor any report", "There is no data or any report."),
                          ("we did not send the deck nor the pack nor the model",
                           "We did not send the deck, the pack or the model."),
                          ("we can neither ship it nor sign it", "We can not ship it or sign it.")):
            with self.subTest(said):
                self.assertIsNone(cleanup.check(said, out, TERMS), out)

    def test_still_refused(self):
        for said, out in (("we can neither ship it nor sign it this week", "We can ship it and sign it this week."),
                          ("we can neither ship it nor sign it this week",
                           "We can neither ship it and sign it this week."),
                          ("Adam did not call us. Nor did Sarah.", "Adam did not call us. Sarah did."),
                          ("we did not send the deck nor the pack", "We did not send the deck and the pack."),
                          ("we did not send the deck nor the pack", "We did not send the deck but the pack."),
                          ("we did not send the deck nor the pack", "We did send the deck nor the pack."),
                          ("we did not send the deck nor the pack", "We did send the deck or the pack."),
                          ("I don't like it nor does Adam", "I don't like it, or does Adam."),
                          ("I will not go. Nor Sarah.", "I will not go, or Sarah."),
                          ("we sent the deck nor the pack", "We sent the deck or the pack."),
                          ("we did send the deck or the pack", "We did not send the deck nor the pack."),
                          ("we did not send the deck we did send the pack", "We did not send the deck nor the pack."),
                          ("adam will call us or sarah will", "Adam will call us, nor will Sarah."),
                          ("we did not send the deck or the pack", "We did not send the deck nor the pack."),
                          ("we did not call and we sent the deck or the pack",
                           "We did not call and we sent the deck nor the pack."),
                          ("we did not call but we sent the deck or the pack",
                           "We did not call, but we sent the deck nor the pack."),
                          ("we did not send the deck, or we sent it late",
                           "We did not send the deck, nor we sent it late.")):
            with self.subTest(out):
                self.assertEqual(cleanup.check(said, out, TERMS), "negation", out)


# ---------------------------------------------------------------- review fixes 5 (31 Oct 2026)

class TestRound3Gaps(unittest.TestCase):
    """Pieces of the round-3 fix no other test pins: each fails on its own mutant."""

    def test_nor_as_the_last_word(self):
        """No other test ends a dictation with a tied "nor" (without the end guard, IndexError)."""
        self.assertIsNone(cleanup.check("we did not call adam nor", "We did not call Adam nor.", TERMS))

    def test_a_line_break_ends_a_sentence(self):
        """A "nor" on a new line negates on its own, as after a full stop."""
        self.assertEqual(cleanup.check("Adam did not call us\nNor Sarah", "Adam did not call us or Sarah.", TERMS),
                         "negation")

    def test_a_corrected_nor_is_not_discounted_twice(self):
        """A tied "nor" a marker takes away is already out of the count the output must keep."""
        said = "we did not send the deck to the client on friday or monday this week nor the pack sorry the model"
        self.assertEqual(cleanup.check(said, "We did send the deck to the client on Friday or Monday this week, or the "
                                             "model.", TERMS), "negation")
        self.assertIsNone(cleanup.check(said, "We did not send the deck to the client on Friday or Monday this week, "
                                              "or the model.", TERMS))

    def test_a_lead_in_after_another_sentence(self):
        """The lead-in is only the text after the last sentence end."""
        self.assertIsNone(cleanup.check(
            "We met Adam today. Can you check these for me, the pack, the deck and the model?",
            "We met Adam today. Can you check the following for me:\n- The pack\n- The deck\n- The model", TERMS))

    def test_free_words_in_a_lead_in(self):
        """A FREE word the model adds to the lead-in ('the following') is not a new word."""
        self.assertIsNone(cleanup.check("Can you check these for me, pack, deck and model?",
                                        "Can you check the following for me:\n- Pack\n- Deck\n- Model", TERMS))


# ---------------------------------------------------------------- live

@unittest.skipUnless(LIVE, "set SCRIBE_LIVE_CLEANUP=1 to run the cases against the local model")
class TestLiveCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = copy.deepcopy(config.DEFAULTS)
        cls.settings.update(owner_name="Alex Morgan", languages=["en", "af"], vocabulary=list(TERMS))
        if os.environ.get("SCRIBE_CLEANUP_MODEL"):
            cls.settings["dictation"]["cleanup"]["model"] = os.environ["SCRIBE_CLEANUP_MODEL"]
        cls.terms = TERMS
        cleanup.Cleaner(cls.settings, terms_fn=lambda: cls.terms).warm(wait=True)

    def test_cases(self):
        for case in CASES["cases"]:
            pairs = [tuple(p) for p in case.get("pairs", [])]
            c = cleanup.Cleaner(self.settings, terms_fn=lambda: self.terms, pairs_fn=lambda p=pairs: p)
            with self.subTest(case["id"]):
                out = c.clean(case["input"])
                fails = evaluate(case, out, c.last["used"])
                self.assertEqual(fails, [], f"{c.last} -> {out!r}")

    def test_warm_latency(self):
        c = cleanup.Cleaner(self.settings, terms_fn=lambda: self.terms)
        base = CASES["latency_input"]
        times = []
        for name in ("Adam", "Sarah", "Mike", "Lindiwe", "Johan"):
            c.clean(base.replace("Adam", name))
            times.append(c.last["ms"])
            self.assertEqual(c.last["reason"], "ok")
        self.assertLess(statistics.median(times), 1200, times)

    def test_no_proxy(self):
        """privacy-failure F6: with HTTP_PROXY and ALL_PROXY set to a stand-in, the local model still answers
        and the stand-in hears nothing."""
        proxy = FakeOllama()
        self.addCleanup(proxy.close)
        c = cleanup.Cleaner(self.settings, terms_fn=lambda: self.terms)
        env = {**without_proxy_settings(), "HTTP_PROXY": proxy.url, "ALL_PROXY": proxy.url}
        with mock.patch.dict(os.environ, env, clear=True):
            c.clean(CASES["latency_input"])
        self.assertEqual(c.last["reason"], "ok")
        self.assertEqual(proxy.requests, [])


if __name__ == "__main__":
    unittest.main()
