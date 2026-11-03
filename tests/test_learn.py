"""learn: the Lexicon file and its rules, the pure edit comparison, and EditWatcher's polling with a fake
UI Automation backend.

Nothing here reads another app. Lexicon files live under tempfile dirs. The live class (SCRIBE_LIVE_UIA=1)
makes its own hidden Win32 EDIT boxes on a thread of its own (never shown, destroyed at the end) and reads them
through real UI Automation: ValuePattern with the caret, the TextPattern window around the caret, IsPassword,
CompareElements, and one whole EditWatcher run whose focus lookup is pointed at the hidden box (the only part
not covered is GetFocusedElement choosing the box, which needs a visible window).
"""
from __future__ import annotations

import codecs
import ctypes
import json
import logging
import os
import sys
import tempfile
import threading
import time
import unittest
from ctypes import wintypes
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scribe import learn  # noqa: E402
from scribe.learn import EditWatcher, Lexicon, learn_pairs  # noqa: E402

logging.getLogger("scribe").addHandler(logging.NullHandler())

P = "the cache flows look fine for wispr flow"


def tag(n: int) -> str:
    """n in letters (12 -> "bc"), for made-up words: a digit with other letters glued to it on each side is a
    figure change ("x1" -> "Xq1"), so it is never learned."""
    return "".join("abcdefghij"[int(d)] for d in str(n))


def lp(before: str, old: str, new: str) -> list[tuple[str, str]]:
    return learn_pairs(before, before.replace(old, new, 1), before)


def wait_for(cond, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.01)
    return bool(cond())


class TempLexicon(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "Scribe" / "lexicon.json"
        self.lex = Lexicon(self.path)

    def add(self, heard, written, times=1, lex=None):
        for _ in range(times):
            (lex or self.lex).add(heard, written)


# ---------------------------------------------------------------- words


class TestWords(unittest.TestCase):
    def test_proper(self):
        for w in ("BlueSky", "ORCA", "Acme", "Wispr Flow", "Help Desk", "iPhone", "Aidan"):
            self.assertTrue(learn.is_proper(w), w)
        for w in ("The", "May", "OK", "cash", "help desk", "I'm", "There"):
            self.assertFalse(learn.is_proper(w), w)

    def test_special(self):
        for w in ("cash", "Whit", "Q3", "n8n", "Wispr Flow", "lekker braai"):
            self.assertTrue(learn.is_special(w), w)
        for w in ("there", "The", "going", "ja", "dit is"):
            self.assertFalse(learn.is_special(w), w)

    def test_common_lists(self):
        self.assertGreater(len(learn.COMMON_WORDS), 400)
        for w in ("the", "was", "were", "with", "yeah", "die", "nie", "baie", "don’t", "Don't"):
            self.assertTrue(learn.is_common(w), w)

    def test_sounds_alike(self):
        for h, w in (("cache", "cash"), ("kiera", "Kira"), ("whisper flow", "Wispr Flow"), ("a den", "Aidan"),
                     ("with", "Whit"), ("yeah", "ja"), ("what", "wat"), ("donkey", "dankie")):
            self.assertTrue(learn.sounds_alike(h, w), (h, w))
        for h, w in (("was", "were"), ("send", "post"), ("report", "numbers"), ("is", "are")):
            self.assertFalse(learn.sounds_alike(h, w), (h, w))

    def test_number_words(self):
        nw = learn.number_words
        for text, want in (("need Fifteen people", ["fifteen"]), ("fifteenth, fiftieth", ["fifteenth", "fiftieth"]),
                           ("Vyf-en-twintig", ["vyf", "twintig"]), ("vyf en twintig", ["vyf", "twintig"]),
                           ("vyfentwintig", ["vyf", "twintig"]), ("drieënveertig", ["drie", "veertig"]),
                           ("tweehonderd rand", ["twee", "honderd"]), ("Twenty-first", ["twenty", "first"]),
                           ("negentienhonderd", ["negentien", "honderd"]), ("die vyftigste keer", ["vyftigste"]),
                           ("millions of them", ["millions"]), ("twenty fifteen", ["twenty", "fifteen"]),
                           ("the ninetieth or hundredth time", ["ninetieth", "hundredth"])):
            self.assertEqual(nw(text), want, text)
        for text, want in (("PE in the high teens", ["teens"]), ("tens of them", ["tens"]),
                           ("the twenties and Nineties", ["twenties", "nineties"]),
                           ("fifteens or fifties", ["fifteens", "fifties"]), ("two thirds", ["two", "thirds"]),
                           ("five sixteenths, five sixths", ["five", "sixteenths", "five", "sixths"]),
                           ("hundredths", ["hundredths"]), ("sixes and sevens", ["sixes", "sevens"]),
                           ("zeros or zeroes", ["zeros", "zeroes"]), ("in sy vyftigs", ["vyftigs"]),
                           ("die sestigs", ["sestigs"]), ("vyfentwintigs", ["vyf", "twintigs"]),
                           ("een biljoen", ["een", "biljoen"]), ("een triljoen", ["een", "triljoen"]),
                           ("triljoene", ["triljoene"])):
            self.assertEqual(nw(text), want, text)     # plurals and fractions: through their singular
        for text in ("often the tent attended", "agter die eenheid", "tienen", "", "R2,000 and 15%", "half past",
                     "the ones we like", "a few seconds", "nie eens nie",     # mostly not numbers
                     "Dries said the paint dries",     # a name and a verb, not drie + s
                     "one" * 2000):              # a word typed into the file far longer than any pair: not split
            self.assertEqual(nw(text), [], text[:20])

    def test_number_words_cover_cleanup(self):
        from scribe import cleanup
        for name in ("_ONES", "_TEENS", "_TENS", "_ORD_ONES", "_ORD_TEENS", "_ORD_TENS", "_HUNDRED", "_SCALES"):
            self.assertLessEqual(set(getattr(cleanup, name)), learn.NUMBER_WORDS, name)
        for w in ("point", "komma", "half"):
            self.assertNotIn(w, learn.NUMBER_WORDS)


# ---------------------------------------------------------------- the comparison


class TestLearnPairs(unittest.TestCase):
    def test_word_fixed(self):
        self.assertEqual(learn_pairs("Note: " + P, "Note: the cash flows look fine for wispr flow", P),
                         [("cache", "cash")])

    def test_name_capitalised(self):
        self.assertEqual(learn_pairs(P, "the cache flows look fine for Wispr Flow", P),
                         [("wispr flow", "Wispr Flow")])

    def test_name_respelled_as_one_phrase(self):
        self.assertEqual(learn_pairs("use whisper flow daily", "use Wispr Flow daily", "use whisper flow daily"),
                         [("whisper flow", "Wispr Flow")])

    def test_two_fixes(self):
        self.assertEqual(learn_pairs(P, "the cash flows look fine for Wispr Flow", P),
                         [("cache", "cash"), ("wispr flow", "Wispr Flow")])

    def test_words_joined_into_a_name(self):
        b = "Send it to blue sky today"
        self.assertEqual(learn_pairs(b, "Send it to BlueSky today", b), [("blue sky", "BlueSky")])

    def test_split_word_to_name(self):
        b = "Ask a den about it."
        self.assertEqual(learn_pairs(b, "Ask Aidan about it.", b), [("a den", "Aidan")])

    def test_afrikaans_sound_alike(self):
        b = "yeah ons moet dit check"
        self.assertEqual(learn_pairs(b, "ja ons moet dit check", b), [("yeah", "ja")])

    def test_sentence_start(self):
        b = "Cache flows look fine."
        self.assertEqual(learn_pairs(b, "Cash flows look fine.", b), [("Cache", "Cash")])
        b = "Yeah ons moet dit check."
        self.assertEqual(learn_pairs(b, "Ja ons moet dit check.", b), [("yeah", "ja")])
        b = "cache flows look fine."
        self.assertEqual(learn_pairs(b, "Cache flows look fine.", b), [])     # a sentence's first capital

    def test_name_at_sentence_start_keeps_its_capital(self):
        self.assertEqual(learn_pairs("Kiera said hi.", "Kira said hi.", "Kiera said hi."), [("Kiera", "Kira")])
        self.assertEqual(learn_pairs("Tim called.", "Kim called.", "Tim called."), [("Tim", "Kim")])

    def test_equal_length_phrase_compared_word_for_word(self):
        b = "kira send it now"
        self.assertEqual(learn_pairs(b, "Kiera sent it now", b), [("kira", "Kiera")])

    def test_word_deleted(self):
        self.assertEqual(learn_pairs(P, "the cache flows look fine for flow", P), [])
        self.assertEqual(learn_pairs(P, "", P), [])

    def test_sentence_rewritten(self):
        self.assertEqual(learn_pairs(P, "numbers are all good on the dashboard today", P), [])

    def test_short_rewrite_with_unlike_words(self):
        self.assertEqual(learn_pairs("Send it.", "Post now.", "Send it."), [])

    def test_text_typed_after(self):
        self.assertEqual(learn_pairs(P, P + " and more words here", P), [])
        self.assertEqual(learn_pairs("the flow", "the flows are good", "the flow"), [])

    def test_edits_outside_the_pasted_span(self):
        self.assertEqual(learn_pairs("Dear kira, " + P, "Dear Kira, " + P, P), [])
        self.assertEqual(learn_pairs(P + " cache it", P + " cash it", P), [])

    def test_pasted_twice_without_caret(self):
        before = P + ". " + P
        self.assertEqual(learn_pairs(before, before.replace("cache", "cash", 1), P), [])

    def test_pasted_twice_caret_picks_the_copy(self):
        before = P + ". " + P
        end = len(before)
        second = P + ". " + P.replace("cache", "cash")
        first = P.replace("cache", "cash") + ". " + P
        self.assertEqual(learn_pairs(before, second, P, caret=end), [("cache", "cash")])
        self.assertEqual(learn_pairs(before, first, P, caret=end), [])
        self.assertEqual(learn_pairs(before, first, P, caret=len(P)), [("cache", "cash")])

    def test_pasted_not_in_box(self):
        self.assertEqual(learn_pairs("something else", "something different", P), [])

    def test_punctuation_and_case_only(self):
        b = "the report is in"
        self.assertEqual(learn_pairs(b, "the report is in.", b), [])
        self.assertEqual(learn_pairs(b, "The report is in", b), [])
        self.assertEqual(learn_pairs("we use Wispr Flow", "we use Wispr flow", "we use Wispr Flow"), [])

    def test_grammar_edits(self):
        for b, a in (("it was fine then", "it were fine then"), ("I will send it", "I will sent it"),
                     ("the flow is ok", "the flows is ok"), ("a apple", "an apple")):
            self.assertEqual(learn_pairs(b, a, b), [], b)

    def test_more_than_three_words(self):
        b = "please call the blue sky team"
        self.assertEqual(learn_pairs(b, "please call Bloo Skie Teem Inc today", b), [])

    def test_common_word_misheard_name(self):
        b = "thanks with for that"
        self.assertEqual(learn_pairs(b, "thanks Whit for that", b), [("with", "Whit")])

    def test_figures_are_never_learned(self):
        for b, a in (("revenue fell 15% this year", "revenue fell 16% this year"), ("In Q3 we grew", "In Q4 we grew"),
                     ("in 2026 we grew", "in 2027 we grew"), ("ask at 10 today", "ask at 11 today"),
                     ("buy 200 shares now", "buy 300 shares now"), ("costs R2,000 each", "costs R2,500 each"),
                     ("rate 12.5% now", "rate 12.8% now"), ("call me at 10", "call me at ten")):
            self.assertEqual(learn_pairs(b, a, b), [], b)
        self.assertEqual(learn_pairs("the q3 numbers", "the Q3 numbers", "the q3 numbers"), [("q3", "Q3")])

    def test_number_words_are_never_learned(self):
        for h, w in (("fifteen", "fifty"), ("thirteen", "thirty"), ("sixteen", "sixty"), ("fourteen", "forty"),
                     ("eighteen", "eighty"), ("nineteen", "ninety"), ("twelve", "twenty"), ("eleven", "seven"),
                     ("fifteenth", "fiftieth"), ("millions", "billions"), ("vyftien", "vyftig"),
                     ("dertien", "dertig"), ("miljoen", "miljard"), ("vyftiende", "vyftigste"),
                     ("tweehonderd", "driehonderd"), ("vyfentwintig", "vyfendertig"),
                     ("drieënveertig", "drieëndertig"), ("twenty-five", "thirty-five"),
                     ("vyf-en-twintig", "vyf-en-dertig"), ("teens", "tens"), ("fifteens", "fifties"),
                     ("sixteenths", "sixths"), ("twenties", "thirties"), ("thirds", "fifths"),
                     ("vyftigs", "sestigs"), ("biljoen", "triljoen"), ("miljoen", "biljoen")):
            b = f"we need {h} people for the launch"
            self.assertEqual(learn_pairs(b, b.replace(h, w), b), [], h)
        b = "send fifteen slides to kiera today"        # an unchanged figure does not stop the fix beside it
        self.assertEqual(learn_pairs(b, b.replace("kiera", "Kira"), b), [("kiera", "Kira")])

    def test_plural_number_words_are_never_learned(self):
        for b, old, new in (("PE in the high teens", "teens", "tens"), ("in the fifteens now", "fifteens", "fifties"),
                            ("five sixteenths of it", "sixteenths", "sixths"),
                            ("die vyftigs was goed", "vyftigs", "sestigs"),
                            ("een biljoen rand", "biljoen", "triljoen"), ("tens of millions", "tens", "teens"),
                            ("in his fifties now", "fifties", "sixties")):
            self.assertEqual(lp(b, old, new), [], old)
        self.assertEqual(lp("the ones we like", "ones", "once"), [("ones", "once")])     # not a number
        self.assertEqual(lp("ask drees about it", "drees", "Dries"), [("drees", "Dries")])    # a name

    def test_long_run_without_a_digit_is_quick(self):
        """A long run of letters the user pastes over a word is read once, not once per letter."""
        start = time.perf_counter()
        self.assertTrue(learn._same_figures("a" * 20000, "a" * 20000))
        self.assertLess(time.perf_counter() - start, 0.5)

    def test_letters_glued_to_a_figure(self):
        """The same digits with other letters glued to them are another figure: R15m is not R15bn."""
        for b, old, new in (("R15m of debt", "R15m", "R15bn"), ("R15mn of debt", "R15mn", "R15bn"),
                            ("about 15k a month", "15k", "15m"), ("in Q3 we grew", "Q3", "Q3s"),
                            ("the 2026 plan", "2026", "2026s"), ("a 5x return", "5x", "5y"),
                            ("costs R2,000 each", "R2,000", "R2 000")):
            self.assertEqual(lp(b, old, new), [], old)
        with tempfile.TemporaryDirectory() as tmp:
            lex = Lexicon(Path(tmp) / "lexicon.json")
            for h, w in (("R1,500", "R1.500"), ("12.5%", "12,5%"), ("r15m", "R15bn"), ("q3", "Q3s")):
                lex.add(h, w)
                lex.add(h, w)
            self.assertEqual(lex.pairs(), [])               # 1,500 is not 1.5
        for b, old, new in (("the q3 numbers", "q3", "Q3"), ("use gpt 4o daily", "gpt 4o", "GPT-4o"),
                            ("in (q3) we grew", "q3", "Q3")):
            self.assertEqual(lp(b, old, new), [(old, new)], old)    # case and punctuation are not figures

    def test_names_joined_from_number_words(self):
        """Number words joined into one name keep the figure: learned (as hints, never applied)."""
        for b, old, new in (("buy ten cent shares today", "ten cent", "Tencent"),
                            ("ask first rand about it", "first rand", "FirstRand"),
                            ("save it to one drive now", "one drive", "OneDrive"),
                            ("buy ten sent shares today", "ten sent", "Tencent"),
                            ("open one note now", "one note", "OneNote"),
                            ("ask first rand bank now", "first rand bank", "FirstRand Bank")):
            self.assertEqual(lp(b, old, new), [(old, new)], old)
        self.assertEqual(lp("save it to Microsoft one drive", "one drive", "OneDrive"),     # the name word beside it
                         [("Microsoft one drive", "Microsoft OneDrive")])
        for b, old, new in (("buy ten cent shares today", "ten cent", "tencent"),      # not a name
                            ("we need fifteen people now", "fifteen people", "Fiftypeople"),   # the number changed
                            ("buy 10 cent shares today", "10 cent", "Tencent"),       # a digit dropped
                            ("ask nine t people", "nine t", "Ninety"),                # the name is a number
                            ("buy ten cent shares today", "ten cent", "Tenmilk")):    # the letters are not close
            self.assertEqual(lp(b, old, new), [], new)

    def test_name_keeps_its_unchanged_words(self):
        for b, a, pair in (
                ("ask the Contoso bank team", "ask the Contoso Bank team", ("Contoso bank", "Contoso Bank")),
                ("on the Help desk now", "on the Help Desk now", ("Help desk", "Help Desk")),
                ("use Morningstar direct daily", "use Morningstar Direct daily",
                 ("Morningstar direct", "Morningstar Direct")),
                ("call Coronation fund managers today", "call Coronation Fund Managers today",
                 ("Coronation fund managers", "Coronation Fund Managers")),
                ("use capital IQ daily", "use Capital IQ daily", ("capital IQ", "Capital IQ")),
                ("the Contoso bang said", "the Contoso Bank said", ("Contoso bang", "Contoso Bank"))):
            self.assertEqual(learn_pairs(b, a, b), [pair], b)
        b = "Ask a den about it."                     # a common word opening the sentence is not part of the name
        self.assertEqual(learn_pairs(b, "Ask Aidan about it.", b), [("a den", "Aidan")])
        b = "Thanks Contoso. kira will call."           # nor is the last word of the sentence before
        self.assertEqual(learn_pairs(b, "Thanks Contoso. Kiera will call.", b), [("kira", "Kiera")])

    def test_coined_spelling(self):
        b = "I bought an I phone yesterday"
        self.assertEqual(learn_pairs(b, "I bought an iPhone yesterday", b), [("I phone", "iPhone")])

    def test_rewrite_is_not_learned(self):
        b = "ask kira to send the deck to aidan today"
        self.assertEqual(learn_pairs(b, "ask Kiera to sent a deck to Aiden tonight", b), [])
        self.assertEqual(learn_pairs(b, "ask Kiera to send the deck to aidan today", b), [("kira", "Kiera")])

    def test_crlf_box(self):
        self.assertEqual(learn_pairs("line one\r\nthe cache flows", "line one\r\nthe cash flows", "the cache flows"),
                         [("cache", "cash")])
        before = "x\r\n" + P + "\r\n" + P
        after = "x\r\n" + P + "\r\n" + P.replace("cache", "cash")
        self.assertEqual(learn_pairs(before, after, P, caret=len(before)), [("cache", "cash")])

    def test_unchanged(self):
        self.assertEqual(learn_pairs(P, P, P), [])
        self.assertEqual(learn_pairs("", "x", P), [])
        self.assertEqual(learn_pairs(P, "x", ""), [])


# ---------------------------------------------------------------- lexicon file


class TestLexiconFile(TempLexicon):
    def test_created_on_first_add(self):
        self.assertFalse(self.path.exists())
        self.assertEqual(self.lex.pairs(), [])
        self.assertEqual(self.lex.apply("the cache"), "the cache")
        self.assertFalse(self.path.exists())
        self.lex.add("cache", "cash")
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual([(e["heard"], e["written"], e["count"]) for e in data["entries"]], [("cache", "cash", 1)])

    def test_default_path(self):
        self.assertEqual(learn.LEXICON_PATH, learn.config.APP_DIR / "lexicon.json")
        with mock.patch.object(learn, "LEXICON_PATH", self.path):      # never reads the user's own lexicon
            self.assertEqual(Lexicon().path, self.path)

    def test_busy_file_is_not_taken_for_corrupt(self):
        self.lex.add("cache", "cash")
        other = Lexicon(self.path)
        self.assertEqual(other.pairs(), [("cache", "cash")])
        data = json.loads(self.path.read_text(encoding="utf-8"))       # another process saves a newer file
        data["entries"].append({"heard": "kiera", "written": "Kira", "count": 1})
        self.path.write_text(json.dumps(data), encoding="utf-8")
        busy = PermissionError(13, "in use")
        with mock.patch.object(Path, "read_text", side_effect=busy), \
                mock.patch.object(Path, "read_bytes", side_effect=busy), \
                self.assertLogs("scribe.learn", "WARNING") as logs:
            self.assertEqual(other.pairs(), [("cache", "cash")])       # kept, not emptied
        self.assertFalse(any("unreadable" in m for m in logs.output))
        self.assertFalse(self.path.with_name("lexicon.json.bad").exists())
        self.assertEqual(sorted(other.pairs()), [("cache", "cash"), ("kiera", "Kira")])   # read again once free

    def test_two_lexicons_on_one_file(self):
        writer, reader = self.lex, Lexicon(self.path)
        for i in range(10):
            writer.add(f"x{tag(i)}", f"Xq{tag(i)}")
        stop = threading.Event()

        def read():
            while not stop.is_set():
                reader.pairs()

        t = threading.Thread(target=read)
        with self.assertNoLogs("scribe.learn", "WARNING"):
            t.start()
            try:
                for i in range(40):
                    writer.add(f"w{tag(i)}", f"Wq{tag(i)}")
                    reader.add(f"r{tag(i)}", f"Rq{tag(i)}")
            finally:
                stop.set()
                t.join()
        self.assertFalse(self.path.with_name("lexicon.json.bad").exists())
        self.assertEqual(len(Lexicon(self.path)), 90)

    def test_persists(self):
        self.add("cache", "cash", 2)
        self.add("Wispr flow", "Wispr Flow")
        again = Lexicon(self.path)
        self.assertEqual(again.pairs(), [("cache", "cash"), ("wispr flow", "Wispr Flow")])
        self.add("cache", "cash", lex=again)
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8"))["entries"][0]["count"], 3)

    def test_failed_replace_keeps_the_old_file(self):
        self.lex.add("cache", "cash")
        old = self.path.read_bytes()
        with mock.patch.object(learn, "_REPLACE_TRIES", 2), \
                mock.patch.object(learn.os, "replace", side_effect=PermissionError(32, "in use")):
            self.lex.add("kiera", "Kira")
        self.assertEqual(self.path.read_bytes(), old)
        self.assertEqual(sorted(p.name for p in self.path.parent.iterdir()), ["lexicon.json"])
        self.assertIn(("kiera", "Kira"), self.lex.pairs())          # kept in memory, saved next time
        self.lex.add("cache", "cash")
        self.assertEqual(len(Lexicon(self.path)), 2)

    def test_readers_never_see_a_partial_file(self):
        stop = threading.Event()
        bad, good = [], [0]

        def reader():
            while not stop.is_set():
                try:
                    json.loads(self.path.read_text(encoding="utf-8"))
                    good[0] += 1
                except (FileNotFoundError, PermissionError):
                    pass
                except ValueError as exc:
                    bad.append(exc)
                time.sleep(0.001)

        t = threading.Thread(target=reader)
        t.start()
        try:
            for i in range(40):
                self.lex.add(f"word{tag(i)}", f"Wordx{tag(i)}")
        finally:
            stop.set()
            t.join()
        self.assertEqual(bad, [])
        self.assertGreater(good[0], 0)
        self.assertEqual(len(Lexicon(self.path)), 40)

    def test_corrupt_file(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("{not json", encoding="utf-8")
        with self.assertLogs("scribe.learn", "WARNING"):
            lex = Lexicon(self.path)
        self.assertEqual(lex.pairs(), [])
        self.assertTrue(self.path.with_name("lexicon.json.bad").exists())
        lex.add("cache", "cash")
        self.assertEqual(Lexicon(self.path).pairs(), [("cache", "cash")])

    ENTRIES = {"entries": [{"heard": "more", "written": "môre", "count": 2},
                           {"heard": "kiera", "written": "Kira", "count": 1}]}

    def write(self, data: bytes):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(data)

    def test_utf16_and_bom_files_are_read(self):
        """A hand edit saved as UTF-16 (Windows PowerShell 5.1's Out-File and >) or UTF-8 with a byte order mark
        keeps its entries; the next save writes plain UTF-8."""
        text = json.dumps(self.ENTRIES, ensure_ascii=False)
        for name, data in (("utf-16", text.encode("utf-16")),
                           ("utf-16-be", codecs.BOM_UTF16_BE + text.encode("utf-16-be")),
                           ("utf-8-sig", text.encode("utf-8-sig"))):
            self.write(data)
            with self.assertNoLogs("scribe.learn", "WARNING"):
                lex = Lexicon(self.path)
                self.assertEqual(sorted(lex.pairs()), [("kiera", "Kira"), ("more", "môre")], name)
            self.assertFalse(self.path.with_name("lexicon.json.bad").exists(), name)
            lex.add("cache", "cash")
            saved = self.path.read_bytes()
            self.assertFalse(saved.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE, codecs.BOM_UTF8)), name)
            self.assertEqual(len(json.loads(saved.decode("utf-8"))["entries"]), 3, name)

    def test_file_in_another_encoding_is_set_aside(self):
        """ANSI (cp1252) with an Afrikaans letter cannot be read: set aside, never raised, so Scribe still starts."""
        self.write(json.dumps(self.ENTRIES, ensure_ascii=False).encode("cp1252"))
        with self.assertLogs("scribe.learn", "WARNING"):
            lex = Lexicon(self.path)
        self.assertEqual(lex.pairs(), [])
        self.assertTrue(self.path.with_name("lexicon.json.bad").exists())
        lex.add("cache", "cash")
        self.assertEqual(Lexicon(self.path).pairs(), [("cache", "cash")])

    def test_file_spoiled_while_running(self):
        """The file goes bad under a running Lexicon: every call still answers, so clean-up keeps working."""
        self.add("cache", "cash", 2)
        self.write(json.dumps(self.ENTRIES, ensure_ascii=False).encode("cp1252"))
        with self.assertLogs("scribe.learn", "WARNING"):
            self.assertEqual(self.lex.pairs(), [])
        self.assertEqual(self.lex.terms(), [])
        self.assertEqual(self.lex.confirmed(), [])
        self.assertEqual(self.lex.apply("the cache"), "the cache")
        self.assertEqual(len(self.lex), 0)
        self.lex.add("kiera", "Kira")
        self.assertEqual(Lexicon(self.path).pairs(), [("kiera", "Kira")])

    def test_counts_that_are_not_finite(self):
        """Infinity, -Infinity, NaN and 1e999 read as a count of 1; the rest of the file is kept."""
        self.write(b'{"entries": [{"heard": "a", "written": "Aq", "count": Infinity},'
                   b' {"heard": "b", "written": "Bq", "count": -Infinity},'
                   b' {"heard": "c", "written": "Cq", "count": NaN},'
                   b' {"heard": "d", "written": "Dq", "count": 1e999},'
                   b' {"heard": "e", "written": "Eq", "count": 3}]}')
        with self.assertNoLogs("scribe.learn", "WARNING"):
            lex = Lexicon(self.path)
            self.assertEqual(len(lex), 5)
        self.assertFalse(self.path.with_name("lexicon.json.bad").exists())
        lex.add("f", "Fq")
        counts = {e["heard"]: e["count"] for e in json.loads(self.path.read_text(encoding="utf-8"))["entries"]}
        self.assertEqual(counts, {"a": 1, "b": 1, "c": 1, "d": 1, "e": 3, "f": 1})

    def test_any_parse_failure_sets_the_file_aside(self):
        """A file nested too deep for the JSON reader, or a parse that overflows, is set aside like bad JSON."""
        self.write(b'{"entries": ' + b"[" * 100000 + b"]" * 100000 + b"}")
        with self.assertLogs("scribe.learn", "WARNING"):
            self.assertEqual(Lexicon(self.path).pairs(), [])
        self.assertTrue(self.path.with_name("lexicon.json.bad").exists())
        self.write(json.dumps(self.ENTRIES).encode("utf-8"))
        with mock.patch.object(learn, "_parse", side_effect=OverflowError("too big")), \
                self.assertLogs("scribe.learn", "WARNING"):
            self.assertEqual(Lexicon(self.path).pairs(), [])

    def test_external_change_and_delete(self):
        other = Lexicon(self.path)
        self.add("cache", "cash", 2)
        self.assertEqual(other.pairs(), [("cache", "cash")])
        self.assertEqual(other.apply("the cache"), "the cash")
        self.path.unlink()
        self.assertEqual(other.pairs(), [])
        self.assertEqual(other.apply("the cache"), "the cache")

    def test_remove(self):
        self.add("cache", "cash")
        self.add("kiera", "Kira")
        self.lex.remove("CACHE")
        self.lex.remove("missing")
        self.assertEqual(Lexicon(self.path).pairs(), [("kiera", "Kira")])

    def test_correction_back_forgets(self):
        self.add("lunch", "launch", 2)
        self.assertEqual(self.lex.apply("lunch at one"), "launch at one")
        self.add("launch", "lunch")
        self.assertEqual(self.lex.pairs(), [])
        self.assertEqual(self.lex.apply("lunch at one"), "lunch at one")

    def test_forgetting_logs_no_words(self):
        self.add("lunch", "launch", 2)
        with self.assertLogs("scribe.learn", "DEBUG") as logs:
            self.add("launch", "lunch")
        self.assertEqual(self.lex.pairs(), [])
        messages = [r.getMessage() for r in logs.records]
        self.assertTrue(any("forgot" in m for m in messages), messages)
        self.assertFalse([m for m in messages if "lunch" in m or "launch" in m])

    def test_case_only_pair_counts_up(self):
        self.add("wispr flow", "Wispr Flow", 3)
        self.assertEqual(self.lex.pairs(), [("wispr flow", "Wispr Flow")])
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8"))["entries"][0]["count"], 3)

    def test_new_written_replaces_the_old(self):
        self.add("kiera", "Kira", 2)
        self.add("kiera", "Kyra")
        self.assertEqual(self.lex.pairs(), [("kiera", "Kyra")])
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8"))["entries"][0]["count"], 1)

    def test_rejected_adds(self):
        for h, w in (("", "x"), ("x", ""), ("same", "same"), ("one two three four", "x"), ("x", "a b c d"),
                     ("x" * 70, "y"), ("...", "!!")):
            self.lex.add(h, w)
        self.assertEqual(self.lex.pairs(), [])
        self.assertFalse(self.path.exists())

    def test_entry_cap(self):
        with mock.patch.object(learn, "MAX_ENTRIES", 5):
            self.add("keep", "Keeper", 3)
            for i in range(8):
                self.lex.add(f"w{tag(i)}", f"Wx{tag(i)}")
            heard = [h for h, _ in self.lex.pairs()]
        self.assertEqual(len(heard), 5)
        self.assertIn("keep", heard)
        self.assertIn(f"w{tag(7)}", heard)
        self.assertNotIn(f"w{tag(0)}", heard)

    def test_threads(self):
        def work(n):
            for i in range(25):
                self.lex.add("cache", "cash")
                self.lex.add(f"t{tag(n)}w{tag(i)}", f"T{tag(n)}x{tag(i)}")

        threads = [threading.Thread(target=work, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        again = Lexicon(self.path)
        self.assertEqual(len(again), 1 + 8 * 25)
        self.assertEqual(again.pairs()[0], ("cache", "cash"))
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8"))["entries"][0]["count"], 200)


# ---------------------------------------------------------------- rules


class TestLexiconRules(TempLexicon):
    def test_terms_ranking(self):
        self.add("cache", "cash", 3)
        self.add("wispr flow", "Wispr Flow")
        self.add("their", "there", 5)         # a common word: not worth a prompt slot
        self.add("with", "Whit", 2)
        self.add("yeah", "ja", 4)
        self.assertEqual(self.lex.terms(), ["cash", "Whit", "Wispr Flow"])

    def test_terms_cap_and_dedupe(self):
        for i in range(60):
            self.add(f"name{tag(i)}", f"Namex{tag(i)}", 1 + i % 3)
        self.add("kiera", "Kira")
        self.add("kyra", "kira")
        terms = self.lex.terms()
        self.assertEqual(len(terms), learn.MAX_TERMS)
        self.assertTrue(all(t in {f"Namex{tag(i)}" for i in range(60) if i % 3 == 2} for t in terms[:20]))
        self.assertEqual(len({t.lower() for t in terms}), len(terms))

    def test_pairs_include_unconfirmed(self):
        self.add("with", "Whit")
        self.add("yeah", "ja")
        self.assertEqual(set(self.lex.pairs()), {("with", "Whit"), ("yeah", "ja")})

    def test_apply_whole_words_any_case(self):
        self.add("cache", "cash", 2)
        self.assertEqual(self.lex.apply("cache cachet CACHE precache cache."), "cash cachet Cash precache cash.")

    def test_apply_keeps_sentence_start_capital(self):
        self.add("cache", "cash", 2)
        self.assertEqual(self.lex.apply("Cache flows. The cache is fine. “Cache” too"),
                         "Cash flows. The cash is fine. “Cash” too")

    def test_apply_phrase(self):
        self.add("wispr flow", "Wispr Flow")
        self.assertEqual(self.lex.apply("use wispr   flow daily, Wispr flow!"), "use Wispr Flow daily, Wispr Flow!")

    def test_apply_longest_first_and_no_chains(self):
        self.add("blue sky", "BlueSky")
        self.add("sky", "Skye")
        self.add("cache", "cash", 2)
        self.add("cash", "money", 2)
        self.assertEqual(self.lex.apply("blue sky and sky"), "BlueSky and Skye")
        self.assertEqual(self.lex.apply("cache and cash"), "cash and money")

    def test_apply_confirmation(self):
        self.add("gonna", "going")             # written side is a common word: needs a second sighting
        self.assertEqual(self.lex.apply("I'm gonna go"), "I'm gonna go")
        self.add("gonna", "going")
        self.assertEqual(self.lex.apply("I'm gonna go"), "I'm going go")

    def test_apply_name_at_once(self):
        self.add("kiera", "Kira")
        self.add("a den", "Aidan")
        self.assertEqual(self.lex.apply("Kiera and a den met"), "Kira and Aidan met")

    def test_apply_case_only_name_at_once(self):
        self.add("help desk", "Help Desk")
        self.assertEqual(self.lex.apply("on the help desk"), "on the Help Desk")

    def test_apply_never_swaps_common_words(self):
        self.add("with", "Whit", 5)
        self.add("yeah", "ja", 5)
        self.add("their", "there", 5)
        self.assertEqual(self.lex.apply("yeah, with their help"), "yeah, with their help")

    def learn(self, before: str, after: str, times: int = 1):
        pairs = learn_pairs(before, after, before)
        self.assertTrue(pairs, before)
        for _ in range(times):
            for heard, written in pairs:
                self.lex.add(heard, written)

    def test_changed_figures_are_not_stored(self):
        for h, w in (("15%", "16%"), ("q3", "Q4"), ("2026", "2027"), ("10", "11"), ("R2,000", "R2,500"),
                     ("10", "ten")):
            self.add(h, w, 2)
        self.assertEqual(self.lex.pairs(), [])
        self.learn("ask Mack at noon", "ask Mark at noon")
        text = "In Q3 2026 revenue fell 15%; ask Mack at 10."
        self.assertEqual(self.lex.apply(text), text)

    def test_changed_number_words_are_not_stored(self):
        for h, w in (("fifteen", "fifty"), ("thirteen", "thirty"), ("vyftien", "vyftig"), ("millions", "billions"),
                     ("fifteenth", "fiftieth"), ("tweehonderd", "driehonderd"), ("vyf en twintig", "vyf en dertig"),
                     ("fifteen", "15"), ("fifty kiera", "fifteen Kira"), ("teens", "tens"), ("fifteens", "fifties"),
                     ("sixteenths", "sixths"), ("vyftigs", "sestigs"), ("biljoen", "triljoen"), ("r15m", "R15bn"),
                     ("15k", "15m"), ("q3", "Q3s"),
                     ("ten", "Tenn"), ("nine t", "Ninety"), ("ten cent", "tencent"), ("10 cent", "Tencent"),
                     ("fifteen people", "Fiftypeople"), ("ten cent", "Tenmilk"),   # not number words joined
                     ("ten cent 2", "Tencent3"), ("ten cent 2", "Tencent2")):
            self.add(h, w, 2)
        self.assertEqual(self.lex.pairs(), [])
        self.assertFalse(self.path.exists())
        self.add("fifteen kiera", "fifteen Kira")        # the same figure on both sides: an ordinary name fix
        self.assertEqual(self.lex.pairs(), [("fifteen kiera", "fifteen Kira")])
        text = "The fee is fifteen percent and fifteen minutes is enough."
        self.assertEqual(self.lex.apply(text), text)

    def test_figure_entries_in_the_file_are_never_used(self):
        """Entries typed into lexicon.json (or kept from before the rule) that change a figure are never applied,
        listed or sent to Whisper's prompt."""
        bad = [("fifteen", "fifty"), ("vyftien", "vyftig"), ("tweehonderd", "driehonderd"), ("millions", "billions"),
               ("fifteenth", "Fiftieth"), ("15%", "16%"), ("teens", "tens"), ("r15m", "R15bn"),
               ("vyftigs", "sestigs")]
        entries = [{"heard": h, "written": w, "count": 5} for h, w in bad]
        entries.append({"heard": "cache", "written": "cash", "count": 2})
        self.path.parent.mkdir(parents=True)
        self.path.write_text(json.dumps({"version": 1, "entries": entries}), encoding="utf-8")
        lex = Lexicon(self.path)
        text = ("The fee is fifteen percent, vyftien rand, tweehonderd millions, the fifteenth, 15%, in cache, "
                "the high teens, R15m, die vyftigs.")
        self.assertEqual(lex.apply(text), text.replace("cache", "cash"))
        self.assertEqual(lex.confirmed(), [("cache", "cash")])
        self.assertEqual(lex.pairs(), [("cache", "cash")])
        self.assertEqual(lex.terms(), ["cash"])

    def test_teens_never_becomes_tens(self):
        for _ in range(2):
            for h, w in lp("PE in the high teens", "teens", "tens") or [("teens", "tens")]:
                self.lex.add(h, w)
        text = "PE in the high teens"
        self.assertEqual(self.lex.apply(text), text)
        self.assertEqual(self.lex.pairs(), [])

    def test_number_names_are_hints_never_applied(self):
        """"ten cent" -> "Tencent" reaches Whisper's prompt and the hints, but apply() and confirmed() never take
        it: "it costs ten cent" is a real amount."""
        self.add("ten cent", "Tencent", 3)
        self.add("first rand", "FirstRand", 3)
        self.add("one drive", "OneDrive")
        self.assertEqual(sorted(self.lex.terms()), ["FirstRand", "OneDrive", "Tencent"])
        self.assertEqual(set(self.lex.pairs()), {("ten cent", "Tencent"), ("first rand", "FirstRand"),
                                                 ("one drive", "OneDrive")})
        self.assertEqual(self.lex.confirmed(), [])
        text = "It costs ten cent, ask first rand, save it to one drive."
        self.assertEqual(self.lex.apply(text), text)
        stored = json.loads(self.path.read_text(encoding="utf-8"))["entries"]
        self.assertEqual(len(stored), 3)
        lex = Lexicon(self.path)                         # read back from the file: the same
        self.assertEqual(sorted(lex.terms()), ["FirstRand", "OneDrive", "Tencent"])
        self.assertEqual(lex.apply(text), text)

    def test_number_name_written_back_is_forgotten(self):
        """Whisper, primed with "Tencent", writes it for an amount; the user's fix back to "ten cent" forgets the name and
        is never stored itself."""
        self.add("ten cent", "Tencent")
        pairs = lp("it costs Tencent each", "Tencent", "ten cent")
        self.assertEqual(pairs, [("Tencent", "ten cent")])
        for h, w in pairs:
            self.lex.add(h, w)
        self.assertEqual(self.lex.pairs(), [])
        self.assertEqual(self.lex.terms(), [])
        self.lex.add("Tencent", "ten cent")              # nothing to forget: nothing stored
        self.lex.add("OneDrive", "one drive")
        self.assertEqual(self.lex.pairs(), [])
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8"))["entries"], [])

    def test_number_name_in_the_file(self):
        entries = [{"heard": "ten cent", "written": "Tencent", "count": 9},
                   {"heard": "fifteen people", "written": "Fiftypeople", "count": 9}]
        self.path.parent.mkdir(parents=True)
        self.path.write_text(json.dumps({"version": 1, "entries": entries}), encoding="utf-8")
        lex = Lexicon(self.path)
        self.assertEqual(lex.pairs(), [("ten cent", "Tencent")])
        self.assertEqual(lex.terms(), ["Tencent"])
        self.assertEqual(lex.confirmed(), [])
        self.assertEqual(lex.apply("ten cent for fifteen people"), "ten cent for fifteen people")

    def test_one_content_edit_is_not_applied(self):
        d = "Quick update for you: the desk review is agreed, Aidan, and Mack and John will send it on."
        for old, new in (("update", "upgrade"), ("desk", "deck"), ("review", "revisit"), ("agreed", "argued"),
                         ("Aidan", "Aiden"), ("Mack", "Mark"), ("John", "Joan")):
            self.learn(d, d.replace(old, new))
        self.assertEqual(len(self.lex.pairs()), 7)                 # all kept, as hints
        vocab = ["Help Desk"]
        text = "Quick update: the Help Desk review is agreed, Aidan. Ask Mack and John."
        self.assertEqual(self.lex.apply(text), text)
        self.assertEqual(self.lex.apply(text, protect=vocab), text)
        self.assertEqual(self.lex.confirmed(), [])

    def test_second_sighting_confirms_but_vocabulary_stays(self):
        self.learn("the desk is ready", "the deck is ready", 2)
        vocab = ["Help Desk"]
        self.assertEqual(self.lex.apply("the desk at the Help Desk", protect=vocab), "the deck at the Help Desk")
        self.assertEqual(self.lex.apply("the desk at the Help Desk"), "the deck at the Help Deck")
        self.assertEqual(self.lex.confirmed(), [("desk", "deck")])

    def test_name_for_a_name_needs_two_sightings(self):
        self.learn("ask Mack now", "ask Mark now")
        self.assertEqual(self.lex.apply("ask Mack today"), "ask Mack today")
        self.learn("tell Mack it is done", "tell Mark it is done")
        self.assertEqual(self.lex.apply("ask Mack today"), "ask Mark today")

    def test_one_capitalised_word_is_never_applied(self):
        self.learn("send it to mark", "send it to Mark", 3)
        self.assertEqual(self.lex.pairs(), [("mark", "Mark")])
        self.assertEqual(self.lex.apply("we mark the book to market daily"), "we mark the book to market daily")
        self.assertIn("Mark", self.lex.terms())

    def test_name_phrase_applies_only_as_a_phrase(self):
        self.learn("ask the Contoso bank team", "ask the Contoso Bank team")
        self.learn("on the Help desk now", "on the Help Desk now")
        self.assertEqual(self.lex.apply("the bank sent it direct to my desk"), "the bank sent it direct to my desk")
        self.assertEqual(self.lex.apply("the contoso bank team at the help desk"),
                         "the Contoso Bank team at the Help Desk")

    def test_name_at_sentence_start(self):
        self.learn("Kiera said hi.", "Kira said hi.")
        self.assertEqual(self.lex.terms(), ["Kira"])
        self.assertEqual(self.lex.pairs(), [("kiera", "Kira")])
        self.learn("Cache flows look fine.", "Cash flows look fine.", 2)
        self.assertEqual(self.lex.apply("Cache flows. The cache is fine."), "Cash flows. The cash is fine.")
        self.learn("Kiera said hi.", "Kira said hi.")
        self.assertEqual(self.lex.apply("ask Kiera. Kiera knows."), "ask Kira. Kira knows.")

    def test_lower_case_sighting_settles_the_case(self):
        self.learn("Cache flows look fine.", "Cash flows look fine.")
        self.assertIn("Cash", self.lex.terms())
        self.learn("the cache flows", "the cash flows")
        self.assertEqual(self.lex.pairs(), [("cache", "cash")])
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8"))["entries"][0]["count"], 2)
        self.assertEqual(self.lex.apply("Cache and cache"), "Cash and cash")

    def test_apply_keeps_a_coined_spelling(self):
        self.learn("I bought an I phone yesterday", "I bought an iPhone yesterday")
        self.learn("check the e bay listing", "check the eBay listing")
        self.assertEqual(self.lex.apply("I bought an I phone yesterday"), "I bought an iPhone yesterday")
        self.assertEqual(self.lex.apply("check the E bay listing"), "check the eBay listing")

    def test_usable(self):
        u = Lexicon.usable
        self.assertTrue(u("wispr flow", "Wispr Flow", 1))
        self.assertFalse(u("may", "May", 9))
        self.assertFalse(u("cache", "cash", 1))       # a word for another word: one edit may be a change of mind
        self.assertTrue(u("cache", "cash", 2))
        self.assertFalse(u("gonna", "going", 1))
        self.assertTrue(u("gonna", "going", 2))
        self.assertFalse(u("with", "Whit", 9))
        self.assertTrue(u("a den", "Aidan", 1))    # ordinary words for a name
        self.assertFalse(u("mack", "Mark", 1, named=True))
        self.assertTrue(u("mack", "Mark", 2, named=True))
        self.assertTrue(u("iphone", "iPhone", 1))
        self.assertTrue(u("q3", "Q3", 1))
        self.assertFalse(u("orca", "ORCA", 1))
        self.assertTrue(u("orca", "ORCA", 2))
        self.assertFalse(u("mark", "Mark", 9))
        self.assertFalse(u("15%", "16%", 9))
        for h, w in (("fifteen", "fifty"), ("vyftien", "vyftig"), ("vyfentwintig", "vyfendertig"),
                     ("millions", "billions"), ("fifteenth", "Fiftieth"), ("fifty kiera", "fifteen Kira"),
                     ("fifteen twenty", "twenty fifteen"),      # the same words in another order: 1520 is not 2015
                     ("teens", "tens"), ("fifteens", "fifties"), ("vyftigs", "sestigs"), ("biljoen", "triljoen"),
                     ("r15m", "R15bn"), ("15k", "15m"), ("q3", "Q3s"),
                     ("ten cent", "Tencent"), ("ten sent", "Tencent"), ("one drive", "OneDrive"),
                     ("first rand", "FirstRand")):              # names joined from number words: never applied
            self.assertFalse(u(h, w, 9), h)
            self.assertFalse(u(h, w, 9, named=True), h)
        self.assertTrue(u("fifteen kiera", "fifteen Kira", 1))


# ---------------------------------------------------------------- watcher


class FakeBox:
    def __init__(self, text: str, caret: int | None = None, password: bool = False, readable: bool = True):
        self.text, self.caret, self.password, self.readable = text, caret, password, readable


class FakeBackend:
    def __init__(self, boxes: dict, focus: str = "box", fail_open: bool = False):
        self.lock = threading.Lock()
        self.boxes, self.focus, self.fail_open = boxes, focus, fail_open
        self.reads = self.opened = self.closed = 0

    def open(self):
        if self.fail_open:
            raise OSError("no UI Automation")
        self.opened += 1

    def close(self):
        self.closed += 1

    def focused(self):
        with self.lock:
            return self.focus

    def same(self, a, b):
        return a == b

    def is_password(self, el):
        return self.boxes[el].password

    def read(self, el, mode=None):
        with self.lock:
            self.reads += 1
            box = self.boxes.get(el)
            if box is None or not box.readable:
                return None
            return box.text, box.caret, "value"

    def set(self, el="box", text=None, focus=None):
        with self.lock:
            if text is not None:
                self.boxes[el].text = text
            if focus is not None:
                self.focus = focus

    def read_count(self):
        with self.lock:
            return self.reads


class TestEditWatcher(TempLexicon):
    def make(self, text, *, poll_s=0.02, settle_s=0.15, max_s=5.0, **box):
        self.backend = FakeBackend({"box": FakeBox(text, **box), "other": FakeBox("")})
        w = EditWatcher(self.lex, poll_s=poll_s, max_s=max_s, settle_s=settle_s, backend=self.backend)
        self.addCleanup(w.stop)
        return w

    def snapped(self):
        self.assertTrue(wait_for(lambda: self.backend.read_count() >= 1))

    def polls(self, n: int):
        start = self.backend.read_count()
        self.assertTrue(wait_for(lambda: self.backend.read_count() >= start + n))

    def test_settled_edit_is_learned(self):
        w = self.make("Hi: " + P)
        w.watch(P)
        self.snapped()
        self.backend.set(text="Hi: " + P.replace("cache", "cash"))
        self.assertTrue(wait_for(lambda: self.lex.pairs() == [("cache", "cash")]))
        self.assertEqual(self.lex.apply("the cache"), "the cache")        # one sighting: not confirmed yet

    def test_focus_moved_uses_the_last_steady_reading(self):
        w = self.make(P, settle_s=30)
        w.watch(P)
        self.snapped()
        self.backend.set(text=P.replace("wispr flow", "Wispr Flow"))
        self.polls(3)
        self.backend.set(text="", focus="other")           # sent: the box empties as focus moves
        self.assertTrue(wait_for(lambda: self.lex.pairs() == [("wispr flow", "Wispr Flow")]))

    def test_half_typed_word_is_not_learned(self):
        w = self.make(P, settle_s=30)
        w.watch(P)
        self.snapped()
        self.backend.set(text=P.replace("cache", "cas"), focus="other")
        self.polls(1)
        time.sleep(0.2)
        reads = self.backend.read_count()
        time.sleep(0.2)
        self.assertEqual(self.backend.read_count(), reads)         # the watch has ended
        self.assertEqual(self.lex.pairs(), [])

    def test_box_cleared_in_place_keeps_the_fix(self):
        w = self.make(P, settle_s=30, max_s=30)
        w.watch(P)
        self.snapped()
        self.backend.set(text=P.replace("cache", "cash"))
        self.polls(3)
        self.backend.set(text="")                           # a chat box sends and keeps focus
        self.assertTrue(wait_for(lambda: self.lex.pairs() == [("cache", "cash")], timeout=2))

    def test_new_watch_ends_the_previous(self):
        w = self.make("A: " + P, settle_s=30, max_s=30)     # only the new watch can end the first one in time
        w.watch(P)
        self.snapped()
        self.backend.set(text="A: " + P.replace("cache", "cash"))
        self.polls(3)
        self.backend.set(text="A: " + P.replace("cache", "cash") + " ask kira today")
        w.watch("ask kira today")
        self.assertTrue(wait_for(lambda: self.lex.pairs() == [("cache", "cash")], timeout=2))
        self.polls(3)
        self.backend.set(text="A: " + P.replace("cache", "cash") + " ask Kira today")
        self.polls(3)
        self.backend.set(focus="other")
        self.assertTrue(wait_for(lambda: ("kira", "Kira") in self.lex.pairs(), timeout=2))

    def test_max_time(self):
        w = self.make(P, settle_s=30, max_s=0.3)
        w.watch(P)
        self.snapped()
        self.backend.set(text=P.replace("cache", "cash"))
        self.assertTrue(wait_for(lambda: self.lex.pairs() == [("cache", "cash")], timeout=3))

    def test_password_box_is_never_read(self):
        w = self.make(P, password=True)
        w.watch(P)
        time.sleep(0.3)
        self.assertEqual(self.backend.read_count(), 0)
        self.assertEqual(self.lex.pairs(), [])

    def test_box_without_patterns(self):
        w = self.make(P, readable=False)
        with self.assertLogs("scribe.learn", "INFO") as logs:
            w.watch(P)
            self.assertTrue(wait_for(lambda: any("skipped" in m for m in logs.output), timeout=3))
        self.assertEqual(self.lex.pairs(), [])
        self.backend.boxes["box"].readable = True              # the thread still serves the next watch
        w.watch(P)
        self.polls(2)
        self.backend.set(text=P.replace("cache", "cash"))
        self.assertTrue(wait_for(lambda: self.lex.pairs() == [("cache", "cash")]))

    def test_paste_that_shows_late(self):
        w = self.make("")
        w.watch(P)
        time.sleep(0.2)
        self.backend.set(text=P)
        self.polls(3)
        self.backend.set(text=P.replace("cache", "cash"))
        self.assertTrue(wait_for(lambda: self.lex.pairs() == [("cache", "cash")]))

    def test_caret_picks_the_copy(self):
        text = P + ". " + P
        w = self.make(text, caret=len(text))
        w.watch(P)
        self.snapped()
        self.backend.set(text=P + ". " + P.replace("cache", "cash"))
        self.assertTrue(wait_for(lambda: self.lex.pairs() == [("cache", "cash")]))

    def test_stop_drops_the_watch(self):
        w = self.make(P, settle_s=30)
        w.watch(P)
        self.snapped()
        self.backend.set(text=P.replace("cache", "cash"))
        self.polls(3)
        w.stop()
        self.assertEqual(self.lex.pairs(), [])
        self.assertIsNone(w._thread)
        self.assertEqual((self.backend.opened, self.backend.closed), (1, 1))

    def test_blank_watch_does_nothing(self):
        w = self.make(P)
        w.watch("   ")
        w.watch("")
        self.assertIsNone(w._thread)

    def test_backend_failure_turns_learning_off(self):
        self.backend = FakeBackend({}, fail_open=True)
        w = EditWatcher(self.lex, poll_s=0.02, backend=self.backend)
        with self.assertLogs("scribe.learn", "ERROR"):
            w.watch(P)
            self.assertTrue(wait_for(lambda: w._broken))
        w.watch(P)
        self.assertTrue(wait_for(lambda: w._thread is None or not w._thread.is_alive()))
        w.stop()

    def test_file_holds_pairs_only(self):
        w = self.make("Private note: " + P)
        w.watch(P)
        self.snapped()
        self.backend.set(text="Private note: " + P.replace("cache", "cash"))
        self.assertTrue(wait_for(lambda: self.lex.pairs() == [("cache", "cash")]))
        stored = self.path.read_text(encoding="utf-8")
        for fragment in ("Private", "flows", "look fine", "wispr"):
            self.assertNotIn(fragment, stored)

    def test_log_holds_counts_only(self):
        """scribe.log outlives a reset of lexicon.json, so a watch logs how many pairs it learned, never which."""
        w = self.make("Note: " + P)
        with self.assertLogs("scribe.learn", "DEBUG") as logs:
            w.watch(P)
            self.snapped()
            self.backend.set(text="Note: " + P.replace("cache", "cash"))
            self.assertTrue(wait_for(lambda: self.lex.pairs() == [("cache", "cash")]))
            self.assertTrue(wait_for(lambda: any(r.levelno == logging.INFO and "1 learned" in r.getMessage()
                                                 for r in logs.records)), logs.output)
        messages = [r.getMessage() for r in logs.records]
        self.assertFalse([m for m in messages if "cache" in m or "cash" in m or "flow" in m], messages)


# ---------------------------------------------------------------- live: real UI Automation, hidden boxes

LRESULT, WPARAM, LPARAM = ctypes.c_ssize_t, ctypes.c_size_t, ctypes.c_ssize_t


class HiddenEdits:
    """A hidden popup holding a multi-line EDIT and a password EDIT, owned and pumped by its own thread.
    No window is ever shown; close() destroys them on that thread."""

    WS_POPUP, WS_CHILD = 0x80000000, 0x40000000
    ES_MULTILINE, ES_PASSWORD, ES_AUTOHSCROLL = 0x0004, 0x0020, 0x0080
    WM_SETTEXT, EM_SETSEL, WM_QUIT = 0x000C, 0x00B1, 0x0012

    def __init__(self):
        u = ctypes.WinDLL("user32", use_last_error=True)
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        u.CreateWindowExW.restype = wintypes.HWND
        u.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
                                      ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                      wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID]
        u.SendMessageW.restype = LRESULT
        u.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, WPARAM, LPARAM]
        u.GetMessageW.restype = wintypes.BOOL
        u.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
        u.TranslateMessage.restype = wintypes.BOOL
        u.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
        u.DispatchMessageW.restype = LRESULT
        u.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
        u.PostThreadMessageW.restype = wintypes.BOOL
        u.PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT, WPARAM, LPARAM]
        u.DestroyWindow.restype = wintypes.BOOL
        u.DestroyWindow.argtypes = [wintypes.HWND]
        u.IsWindowVisible.restype = wintypes.BOOL
        u.IsWindowVisible.argtypes = [wintypes.HWND]
        k.GetCurrentThreadId.restype = wintypes.DWORD
        k.GetCurrentThreadId.argtypes = []
        self.u, self.k = u, k
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, name="test-hidden-edits", daemon=True)
        self.thread.start()
        if not self.ready.wait(5) or not self.edit:
            raise RuntimeError("hidden edit boxes not created")

    def _run(self):
        u = self.u
        self.tid = self.k.GetCurrentThreadId()
        self.parent = u.CreateWindowExW(0, "STATIC", "scribe-test", self.WS_POPUP, 0, 0, 10, 10,
                                        None, None, None, None)
        self.edit = u.CreateWindowExW(0, "EDIT", None, self.WS_CHILD | self.ES_MULTILINE | self.ES_AUTOHSCROLL,
                                      0, 0, 10, 10, self.parent, None, None, None)
        self.password = u.CreateWindowExW(0, "EDIT", None, self.WS_CHILD | self.ES_PASSWORD, 0, 0, 10, 10,
                                          self.parent, None, None, None)
        self.ready.set()
        msg = wintypes.MSG()
        while u.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            u.TranslateMessage(ctypes.byref(msg))
            u.DispatchMessageW(ctypes.byref(msg))
        u.DestroyWindow(self.parent)

    def set_text(self, text: str, caret: int | None = None):
        buf = ctypes.create_unicode_buffer(text)
        self.u.SendMessageW(self.edit, self.WM_SETTEXT, 0, ctypes.cast(buf, ctypes.c_void_p).value)
        if caret is not None:
            self.u.SendMessageW(self.edit, self.EM_SETSEL, caret, caret)

    def visible(self) -> bool:
        return bool(self.u.IsWindowVisible(self.parent) or self.u.IsWindowVisible(self.edit))

    def close(self):
        self.u.PostThreadMessageW(self.tid, self.WM_QUIT, 0, 0)
        self.thread.join(5)


def in_thread(fn):
    """Run fn on a fresh thread (its own COM apartment, like the watcher's) and return or raise its result."""
    out: dict = {}

    def run():
        try:
            out["value"] = fn()
        except BaseException as exc:    # handed back to the test
            out["error"] = exc

    t = threading.Thread(target=run)
    t.start()
    t.join(60)
    if "error" in out:
        raise out["error"]
    return out.get("value")


@unittest.skipUnless(os.environ.get("SCRIBE_LIVE_UIA") == "1",
                     "set SCRIBE_LIVE_UIA=1 to read hidden Win32 edit boxes through UI Automation")
class TestLiveUIA(TempLexicon):
    def setUp(self):
        super().setUp()
        self.boxes = HiddenEdits()
        self.addCleanup(self.boxes.close)

    def test_read_paths(self):
        text = "Hello there. whisper flow is great\r\nline two"
        self.boxes.set_text(text, caret=12)

        def check():
            b = learn.UIABackend()
            b.open()
            try:
                el = b.from_handle(self.boxes.edit)
                pw = b.from_handle(self.boxes.password)
                b.focused()                                  # must not raise; its element is not read
                return (b.is_password(el), b.is_password(pw), b.read(el), b.read(el, "text"),
                        b.same(el, b.from_handle(self.boxes.edit)), b.same(el, pw))
            finally:
                el = pw = None
                b.close()

        is_pw, pw_is_pw, value, window, same, different = in_thread(check)
        self.assertFalse(is_pw)
        self.assertTrue(pw_is_pw)
        self.assertEqual(value, (text, 12, "value"))
        self.assertEqual(window, (text, 12, "text"))
        self.assertTrue(same)
        self.assertFalse(different)
        self.assertFalse(self.boxes.visible())

    def test_long_box_reads_a_window_around_the_caret(self):
        text = "".join(f"w{i:03d} " for i in range(30))          # 150 characters, no line breaks
        self.boxes.set_text(text, caret=60)

        def check():
            b = learn.UIABackend()
            b.open()
            try:
                with mock.patch.object(learn, "MAX_CHARS", 40):
                    return b.read(b.from_handle(self.boxes.edit))
            finally:
                b.close()

        window, caret, mode = in_thread(check)
        self.assertEqual(mode, "text")
        self.assertEqual((window, caret), (text[40:80], 20))

    def test_watcher_learns_from_a_real_box(self):
        boxes = self.boxes

        class PointedAtTheBox(learn.UIABackend):
            def focused(self):
                return self.from_handle(boxes.edit)

        boxes.set_text("Note: the cache flows\r\nline two", caret=len("Note: the cache flows"))
        w = EditWatcher(self.lex, poll_s=0.1, settle_s=0.4, max_s=20, backend=PointedAtTheBox())
        self.addCleanup(w.stop)
        w.watch("the cache flows")
        time.sleep(0.5)
        boxes.set_text("Note: the cash flows\r\nline two")
        self.assertTrue(wait_for(lambda: self.lex.pairs() == [("cache", "cash")], timeout=10))
        w.stop()
        self.assertFalse(boxes.visible())


if __name__ == "__main__":
    unittest.main()
