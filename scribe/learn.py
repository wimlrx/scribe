"""Learn from the user's own corrections, like Wispr Flow's auto-dictionary.

Right after a dictation is pasted, EditWatcher takes the focused text box through UI Automation and keeps a
snapshot of its text in memory. Once the user has edited the box and paused, moved on, or two minutes have passed,
it compares the box with the snapshot once. Where a word or short phrase Scribe typed was replaced by one that
plausibly sounds the same, the pair (what Scribe heard, what the user wrote) goes into the Lexicon at
%LOCALAPPDATA%\\Scribe\\lexicon.json. Snapshots stay in memory and are dropped after the comparison; only the
learned pairs are stored, and only their count is logged. Password boxes are never read.

The Lexicon feeds three things: terms() biases Whisper's prompt towards the user's names and spellings, apply()
respells confirmed pairs in the next dictation (confirmed() lists them), and pairs() lists every learned pair
as a hint for a step that sees the sentence.

What is learned:
- An edit that changes a figure is never learned: 15% -> 16%, Q3 -> Q4, 2026 -> 2027, R15m -> R15bn,
  fifteen -> fifty, teens -> tens and vyftien -> vyftig are new figures, not mishearings. Both sides must hold
  the same digits, each with the same letters glued to it, and the same number words (NUMBER_WORDS, with their
  plurals and fractions), in the same order.
- Number words joined into one name ("ten cent" -> "Tencent", "first rand" -> "FirstRand") are learned for
  terms() and pairs(), but apply() never uses them: "it costs ten cent" is a real amount.
- A name keeps the unchanged name words next to it: "Contoso bank" -> "Contoso Bank" is learned whole, never
  "bank" -> "Bank".
- A sentence's first capital is not learned, and a common word opening a sentence is learned in lower case
  ("Yeah" -> "Ja" teaches "yeah" -> "ja"). Any other capitalised word keeps its capital ("Kiera" -> "Kira").

Confirmation rule:
- A written side is *special* when it looks like a proper noun (a capital after the first letter, a capitalised
  word that is not a common word, or two or more capitalised words), holds a digit, or holds a word outside
  COMMON_WORDS (an unusual spelling). A special written side joins terms() at once, most used first.
- apply() never uses a pair whose heard side is made only of common words ("with" -> "Whit", "yeah" -> "ja"):
  swapping a common word everywhere would be wrong in most sentences.
- Case or spacing only: a name phrase or a coined spelling is used at once ("wispr flow" -> "Wispr Flow",
  "blue sky" -> "BlueSky", "iphone" -> "iPhone", "q3" -> "Q3"); one word in capitals ("orca" -> "ORCA")
  after two sightings; one word gaining a first capital ("mark" -> "Mark") never, since whether it is a name
  depends on the sentence. Whisper's prompt (terms()) carries it instead.
- Ordinary words Scribe wrote where the user meant a name are respelled at once ("a den" -> "Aidan").
- Everything else waits for a second sighting: a word for another word ("cache" -> "cash", "desk" -> "deck"),
  where one edit may be a change of mind, and one name for another ("Mack" -> "Mark"). A pair whose changed
  words Scribe wrote with a capital (a name, or a sentence start) is *named*; a named single-capital pair
  ("Cache" -> "Cash", "Kiera" -> "Kira") follows the dictation's capital in apply(): "the cache" -> "the cash".
- apply() keeps a capital Whisper wrote where the learned form starts lower case, unless the learned form is a
  coined spelling ("I phone" -> "iPhone"), and leaves the configured vocabulary it is given alone.
- Correcting a learned respelling back (Scribe wrote "launch", the user types "lunch" again) forgets that pair.
"""
from __future__ import annotations

import codecs
import difflib
import functools
import gc
import json
import logging
import math
import os
import re
import sys
import threading
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Iterable

from . import config

log = logging.getLogger("scribe.learn")

LEXICON_PATH = config.APP_DIR / "lexicon.json"
MAX_WORDS = 3                 # each side of a learned pair
MAX_PAIR_CHARS = 64
MAX_ENTRIES = 2000            # beyond this the least used, oldest entries go
MAX_TERMS = 40                # Whisper's prompt is short; the base vocabulary comes first
MAX_CHARS = 20000             # a text box longer than this is read as a window around the caret
REWRITE_WORDS = 3             # more changed words than this, and over half the dictation: a rewrite
CHAR_RATIO = 0.45             # spelling similarity that counts as the same sound
SOUND_RATIO = 0.7             # similarity of the rough sound keys
JOIN_RATIO = 0.8              # spelling similarity of number words joined into a name ("ten sent" -> "Tencent")
CONFIRM_COUNT = 2
FIND_WAIT_S = 1.0             # how long the pasted text may take to show in the box
FIND_POLL_S = 0.1
STOP_JOIN_S = 5.0
UIA_TIMEOUT_MS = 2000         # a hung app cannot hold the watcher longer than this per call
_REPLACE_TRIES = 10
_READ_TRIES = 5               # a read refused while another process replaces the file is tried again
_READ_WAIT_S = 0.02
_PRONOUN_I = frozenset({"i", "i'm", "i'd", "i'll", "i've"})

_EDGE = "\"'`“”‘’«»()[]{}<>.,;:!?…*_~-–—/\\|"
_TOKEN = re.compile(r"\S+")
_SENT_END = ".!?…\n"
_OPENERS = " \t\"'([{“‘«"

COMMON_WORDS = frozenset("""
a about above across actually after again against ago ahead all almost along already also although always am
among an and another any anyone anything anyway are aren't around as ask asked at away back bad be became
because become been before behind being below best better between big bit both bring brought but by call called
came can can't cannot case certain change clear close come comes coming could couldn't day days did didn't
different do does doesn't doing don't done down during each early easy either else end enough even ever every
everyone everything far feel felt few find first five for found four free from front full gave get gets getting
give given go goes going gone good got great had hadn't half hard has hasn't have haven't having he he's head
hear heard help her here hers herself hey hi high him himself his hold home how however i i'd i'll i'm i've if
important in into is isn't it it's its itself just keep kept kind knew know known large last late later least
leave left less let let's life like line little long look looked lot lots low made main major make makes making
man many may maybe me mean meant men might mind mine more most much must my myself name need needs never new
next nice night no nobody none nor not nothing now number of off often oh ok okay old on once one only open or
other others our ours out over own part people perhaps place please point possible put quite ran rather real
really right run said same saw say says second see seem seemed seen send sent set she she's short should
shouldn't show side simple since small so some someone something sometimes soon sorry sort start still strong
such sure take taken talk tell ten than thank thanks that that's the their theirs them themselves then there
there's these they they're thing things think third this those though thought three through time times to today
together told tomorrow too took top true try trying turn two under until up upon us use used very want wanted
was wasn't way we we'll we're we've week well went were weren't what what's when where whether which while
white who whole whom whose why will with within without won't word words work would wouldn't wrong yeah year
years yes yesterday yet you you'd you'll you're you've young your yours yourself
six seven eight nine hundred thousand million billion
monday tuesday wednesday thursday friday saturday sunday january february march april june july august
september october november december
area black blue book business car child children city client clients company cost country data door email eye
eyes face fact father file friend friends girl government group guy hand hands health history hour hours house
idea information issue job law level market meeting minute minutes moment money month morning mother office
party person plan power price problem program question reason red report research result room school service
story system team value war water woman women world
die en is in van het nie te dat op vir wat met ek sy hy ons hulle jy julle u dit n 'n om aan as ook sal kan moet
wil was word by uit so maar of nog al net baie meer na toe dan daar hier nou reeds weer wees gaan kom doen sê se
gee kry maak sien weet dink hê my jou haar hom hul wie waar wanneer hoe hoekom waarom ja nee dankie asseblief
goed reg lekker groot klein nuwe ou eerste een twee drie vier vyf ses sewe agt nege tien almal iets niks alles
iemand niemand elke ander self tot oor onder agter voor langs tussen deur sonder teen sodat omdat want terwyl
indien tog dus egter selfs miskien seker altyd nooit dikwels soms vandag môre more gister jaar dag maand tyd mens
mense werk huis geld ding dinge regtig darem sommer mos nogal hierdie daardie dié daai watter dis gewees gehad
gesê gedoen gemaak gekry gesien gegaan gekom moes kon wou sou hoef laat
""".split())

# Afrikaans words and the English words Whisper writes for them in the user's speech (and back).
SOUND_ALIKES = frozenset(frozenset(p) for p in (
    ("ja", "yeah"), ("ja", "yah"), ("ja", "ya"), ("nee", "nay"), ("nee", "near"), ("dit", "did"),
    ("dis", "this"), ("die", "the"), ("die", "dee"), ("en", "and"), ("ek", "egg"), ("ek", "ache"),
    ("wat", "what"), ("het", "hat"), ("het", "head"), ("maar", "ma"), ("maar", "mar"), ("ons", "once"),
    ("jy", "yay"), ("hy", "hay"), ("sy", "say"), ("nou", "no"), ("nou", "now"), ("kom", "come"),
    ("gaan", "gun"), ("doen", "done"), ("ook", "oak"), ("vir", "fear"), ("vir", "fur"), ("van", "fun"),
    ("van", "fan"), ("oor", "or"), ("daar", "dar"), ("baie", "buy a"), ("baie", "bye a"),
    ("dankie", "donkey"), ("goed", "good"), ("lekker", "lacquer"), ("sal", "sell"), ("wil", "will"),
    ("kan", "con"), ("hulle", "hula"), ("weer", "where"), ("reg", "rag"), ("moet", "mood"),
    ("eers", "ears"), ("seker", "sicker"), ("mos", "moss"), ("sommer", "summer"), ("ag", "ugh"),
))
_SOUND_RULES = (("sch", "sk"), ("tch", "ch"), ("ph", "f"), ("wh", "w"), ("ck", "k"), ("qu", "kw"),
                ("gh", "g"), ("th", "t"), ("dt", "t"), ("ce", "se"), ("ci", "si"), ("cy", "si"), ("c", "k"),
                ("x", "ks"), ("z", "s"), ("v", "f"), ("w", "v"))
_INFLECTIONS = frozenset({"", "s", "es", "d", "ed", "ing", "'s", "s'", "n", "en", "e", "er", "est", "ly", "t"})

# Number words, English and Afrikaans, cardinal and ordinal: a pair that changes one changes a figure, like a
# changed digit. Every word of cleanup's number tables (without point, komma and half), plus the ordinals and
# plural scales those leave out. Afrikaans writes compounds as one word; number_words() splits them. A plural or
# fraction counts through its singular (tens, twenties, fifteens, sixths, sixteenths, vyftigs), except the few
# in _NOT_PLURALS, which mostly are not numbers.
NUMBER_WORDS = frozenset("""
zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen
eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion trillion
teens hundreds thousands millions billions trillions
first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth
sixteenth seventeenth eighteenth nineteenth twentieth thirtieth fortieth fiftieth sixtieth seventieth eightieth
ninetieth hundredth thousandth millionth billionth trillionth
nul een twee drie vier vyf ses sewe agt nege tien elf twaalf dertien veertien vyftien sestien sewentien agtien
negentien twintig dertig veertig vyftig sestig sewentig tagtig negentig honderd duisend miljoen miljard
biljoen biljard triljoen
honderde duisende miljoene miljarde biljoene biljarde triljoene
eerste tweede derde vierde vyfde sesde sewende agtste negende tiende elfde twaalfde dertiende veertiende
vyftiende sestiende sewentiende agtiende negentiende twintigste dertigste veertigste vyftigste sestigste
sewentigste tagtigste negentigste honderdste duisendste miljoenste miljardste biljoenste biljardste triljoenste
""".split())
_PLURALS = (("ies", "y"), ("es", ""), ("s", ""))        # twenties -> twenty, sixes -> six, sixths -> sixth
_NOT_PLURALS = frozenset({"ones", "seconds", "eens", "dries"})   # the ones, a few seconds, nie eens nie, Dries
_NUMBER_JOINS = ("", "en", "ën")      # between the parts of a compound: tweehonderd, vyfentwintig, drieënveertig
_NUMBER_LEN = max(len(w) for w in NUMBER_WORDS)
_LETTERS = re.compile(r"[^\W\d_]+")
# Letters and digits run together (R15m, Q3, R2,000, 12.5), matched only from a run's start: a long run with no
# digit costs one pass, not one per letter.
_FIGURE = re.compile(r"(?<![^\W_])[^\W_]*\d(?:[.,]?\d)*[^\W_]*")


# ---------------------------------------------------------------- words

def _strip(tok: str) -> str:
    return tok.strip(_EDGE)


def _words(text: str) -> list[str]:
    return [w for w in (_strip(t) for t in text.split()) if w]


def _key(text: str) -> str:
    """Lower case, edge punctuation off, single spaces: how a heard side is stored and matched."""
    return " ".join(_words(text.replace("’", "'"))).lower()


def _display(text: str) -> str:
    return " ".join(_words(text))


def _core(text: str) -> str:
    """Letters and digits only, lower case: what is left once case, spacing and punctuation are ignored."""
    return "".join(c for c in text.lower() if c.isalnum())


def is_common(word: str) -> bool:
    return _strip(word).replace("’", "'").lower() in COMMON_WORDS


def is_proper(written: str) -> bool:
    """Looks like a proper noun: BlueSky, ORCA, Acme, Help Desk. A lone common word with a capital
    first letter ("The", "May") does not count: that is a sentence start, not a name."""
    words = _words(written)
    if not words:
        return False
    for w in words:
        if any(c.isupper() for c in w[1:]) and not (w.isupper() and is_common(w)):
            return True
        if w[:1].isupper() and not is_common(w):
            return True
    return len(words) >= 2 and all(w[:1].isupper() for w in words)


def is_special(written: str) -> bool:
    """Worth biasing Whisper towards: a proper noun, a digit, or a word outside the common list."""
    words = _words(written)
    return bool(words) and (is_proper(written) or any(c.isdigit() for c in written)
                            or any(not is_common(w) for w in words))


def _digits(text: str) -> str:
    return "".join(c for c in text if c.isdigit())


@functools.lru_cache(maxsize=4096)
def _number_parts(word: str) -> tuple[str, ...]:
    """The number words one lower-case word is made of: itself (a plural or fraction too: tens, sixths), or the
    parts of a compound written as one word (tweehonderd -> twee honderd, vyfentwintig -> vyf twintig). Empty
    when it is not a number."""
    if word in NUMBER_WORDS or (word not in _NOT_PLURALS and any(
            word.endswith(end) and word[:-len(end)] + base in NUMBER_WORDS for end, base in _PLURALS)):
        return (word,)
    if len(word) > MAX_PAIR_CHARS:
        return ()
    for i in range(min(len(word) - 1, _NUMBER_LEN), 2, -1):        # the longest first part first
        if word[:i] in NUMBER_WORDS:
            rest = word[i:]
            for join in _NUMBER_JOINS:
                if rest.startswith(join):
                    tail = _number_parts(rest[len(join):])
                    if tail:
                        return (word[:i],) + tail
    return ()


def number_words(text: str) -> list[str]:
    """The number words in a text, in order and in lower case, compounds split: "Vyf-en-twintig",
    "vyf en twintig" and "vyfentwintig" all give ["vyf", "twintig"]."""
    return [p for w in _LETTERS.findall(text.lower()) for p in _number_parts(w)]


def _figure_tokens(text: str) -> list[str]:
    """Each run of letters and digits that holds a digit, with a , or . between its digits, in lower case and
    without the punctuation around it: "R15m" and "R15bn" are two figures although their digits are the same,
    and so are "1,500" and "1.500"."""
    return _FIGURE.findall(text.lower())


def _same_figures(heard: str, written: str) -> bool:
    """Both sides hold the same digits, each with the same letters glued to it, and the same number words:
    anything else is a new figure, not a mishearing (15% -> 16%, R15m -> R15bn, Q3 -> Q3s, fifteen -> fifty,
    teens -> tens, vyftien -> vyftig). Case is not a figure: q3 -> Q3 keeps it."""
    return (_digits(heard) == _digits(written) and _figure_tokens(heard) == _figure_tokens(written)
            and number_words(heard) == number_words(written))


def _number_name(heard: str, written: str) -> bool:
    """Number words joined into one name: "ten cent" -> "Tencent", "one drive" -> "OneDrive", "first rand" ->
    "FirstRand", "ten sent" -> "Tencent". Past the words both sides share at the edges, two or more heard words
    holding a number word become one written word that is a name and no number; it keeps the heard number words
    letter for letter, in order, and its letters match theirs closely. The digits are the same on both sides. The
    figure is written into a name, not changed, so the pair is learned; usable() refuses it all the same, since
    "it costs ten cent" is a real amount."""
    if _digits(heard) != _digits(written) or _figure_tokens(heard) != _figure_tokens(written):
        return False
    h, w = _words(heard), _words(written)
    while h and w and h[0].lower() == w[0].lower():
        h, w = h[1:], w[1:]
    while h and w and h[-1].lower() == w[-1].lower():
        h, w = h[:-1], w[:-1]
    if len(h) < 2 or len(w) != 1 or number_words(w[0]) or not is_proper(w[0]):
        return False
    numbers, name, at = number_words(" ".join(h)), w[0].lower(), 0
    for n in numbers:
        at = name.find(n, at)
        if at < 0:
            return False
        at += len(n)
    return bool(numbers) and difflib.SequenceMatcher(
        None, _core(" ".join(h)), _core(name), autojunk=False).ratio() >= JOIN_RATIO


@functools.lru_cache(maxsize=4096)
def _figures_kept(heard: str, written: str) -> bool:
    """A pair that may be learned and listed: the same figures, or number words joined into a name. Cached:
    pairs() and terms() ask it of every entry on every dictation."""
    return _same_figures(heard, written) or _number_name(heard, written)


def _inner_capital(text: str) -> bool:
    return any(c.isupper() for c in text[1:])


def _coined(written: str) -> bool:
    """Spelt as no dictionary word is: a capital inside a word that has small letters (iPhone, BlueSky), or a
    digit (Q3)."""
    return any(c.isdigit() for c in written) or any(
        _inner_capital(w) and not w.isupper() for w in _words(written))


def _named(heard: str, written: str) -> bool:
    """Scribe wrote the words the user changed with a capital: a name Whisper knew, or a sentence start."""
    kept = set(_words(written))
    changed = [w for w in _words(heard) if w not in kept]
    return bool(changed) and is_proper(" ".join(changed))


def _name_part(word: str, at_start: bool) -> bool:
    """A word that belongs to the name next to it: capitalised, and not a common word opening a sentence."""
    return word[:1].isupper() and _key(word) not in _PRONOUN_I and not (at_start and is_common(word))


def _sound(core: str) -> str:
    """A rough sound key: spelling variants and Afrikaans/English letter values folded, vowels merged."""
    s = "".join(c for c in unicodedata.normalize("NFKD", core) if c.isalpha() and not unicodedata.combining(c))
    for a, b in _SOUND_RULES:
        s = s.replace(a, b)
    s = re.sub(r"[aeiouy]+", "a", s)
    return re.sub(r"(.)\1+", r"\1", s)


def sounds_alike(heard: str, written: str) -> bool:
    """Plausibly the same spoken sound: close spelling, close sound keys, or a known Afrikaans/English pair."""
    if frozenset((_key(heard), _key(written))) in SOUND_ALIKES:
        return True
    ch, cw = _core(heard), _core(written)
    if not ch or not cw:
        return False
    if difflib.SequenceMatcher(None, ch, cw, autojunk=False).ratio() >= CHAR_RATIO:
        return True
    sh, sw = _sound(ch), _sound(cw)
    return bool(sh and sw) and difflib.SequenceMatcher(None, sh, sw, autojunk=False).ratio() >= SOUND_RATIO


def _edge_change(ch: str, cw: str) -> bool:
    """One side is the other with letters added at an edge (flow -> flows, Kira -> Kiras), or the two differ
    only in an inflection ending (send -> sent): grammar or typing on, not a mishearing."""
    if ch.startswith(cw) or cw.startswith(ch) or ch.endswith(cw) or cw.endswith(ch):
        return True
    n = 0
    for a, b in zip(ch, cw):
        if a != b:
            break
        n += 1
    return n >= 3 and ch[n:] in _INFLECTIONS and cw[n:] in _INFLECTIONS


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def _pair(heard: str, written: str, at_start: bool = False) -> list[tuple[str, str]]:
    if heard == written:
        return []
    ch, cw = _core(heard), _core(written)
    if not ch or not cw:
        return []
    if not _figures_kept(heard, written):
        # A changed figure is a new figure, not a mishearing. A name written back as its number words
        # ("Tencent" -> "ten cent", an amount) goes on only so that add() forgets the name; add() never stores it.
        return [(heard, written)] if _number_name(written, heard) else []
    if at_start and written[:1].isupper() and not _inner_capital(written) \
            and all(is_common(w) for w in _words(written)):
        # A common word opening a sentence: "Yeah" -> "Ja" teaches "yeah" -> "ja".
        heard, written = _lower_first(heard), _lower_first(written)
    if ch == cw:
        # Only case, spacing or punctuation changed: learnt only as a proper noun gaining capitals or joining up,
        # never as a sentence's first capital.
        if at_start and heard[1:] == written[1:]:
            return []
        more_caps = sum(c.isupper() for c in written) > sum(c.isupper() for c in heard)
        joined = len(written.split()) != len(heard.split())
        return [(heard, written)] if is_proper(written) and (more_caps or joined) else []
    if _edge_change(ch, cw) or not sounds_alike(heard, written):
        return []
    return [(heard, written)]


def _pairs_from(heard_toks: list[str], written_toks: list[str],
                starts: list[bool] | None = None) -> list[tuple[str, str]]:
    """Pairs from one replaced run of tokens, given with a few unchanged tokens either side and whether each
    heard token opens a sentence. Unchanged words are trimmed off, except name words next to a name being
    written ("Contoso bank" -> "Contoso Bank" is learned whole, never "bank" -> "Bank")."""
    starts = starts or [False] * len(heard_toks)
    h = [(x, s) for x, s in ((_strip(t), s) for t, s in zip(heard_toks, starts)) if x]
    w = [x for x in (_strip(t) for t in written_toks) if x]
    lead = 0
    while lead < len(h) and lead < len(w) and h[lead][0] == w[lead]:
        lead += 1
    trail = 0
    while trail < len(h) - lead and trail < len(w) - lead and h[-1 - trail][0] == w[-1 - trail]:
        trail += 1
    hi, wi = len(h) - trail, len(w) - trail
    if lead >= hi or lead >= wi:
        return []
    if all(x[:1].isupper() for x in w[lead:wi]):         # never across a sentence end
        while lead and not h[lead][1] and _name_part(*h[lead - 1]) and max(hi, wi) - lead < MAX_WORDS:
            lead -= 1
        while trail and not h[hi][1] and _name_part(*h[hi]) and max(hi, wi) - lead < MAX_WORDS:
            hi, wi, trail = hi + 1, wi + 1, trail - 1
    hs, ws = [x for x, _ in h[lead:hi]], w[lead:wi]
    if len(hs) > MAX_WORDS or len(ws) > MAX_WORDS:
        return []
    if len(hs) == len(ws) >= 2 and not all(x[:1].isupper() for x in ws):
        # Word for word, unless the new words read as one name ("whisper flow" -> "Wispr Flow").
        return [p for i, (a, b) in enumerate(zip(hs, ws)) for p in _pair(a, b, h[lead + i][1])]
    return _pair(" ".join(hs), " ".join(ws), h[lead][1])


# ---------------------------------------------------------------- the comparison

def _normalise(text: str) -> str:
    """Line ends and no-break spaces as plain text; edit controls hand back \\r\\n."""
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")


def locate(text: str, pasted: str, caret: int | None = None) -> tuple[int, int] | None:
    """Where the pasted text sits in the box. Found more than once, the copy that ends nearest the caret;
    with no caret to tell them apart, None (an edit could belong to the user's own copy)."""
    if not pasted:
        return None
    starts, i = [], text.find(pasted)
    while i >= 0:
        starts.append(i)
        i = text.find(pasted, i + 1)
    if not starts:
        return None
    if len(starts) > 1:
        if caret is None:
            return None
        starts.sort(key=lambda s: abs(s + len(pasted) - caret))
    return starts[0], starts[0] + len(pasted)


def learn_pairs(before: str, after: str, pasted: str, *, caret: int | None = None) -> list[tuple[str, str]]:
    """(heard, written) pairs from one edit of a pasted dictation.

    `before` is the box right after the paste, `after` once the user has edited it, `caret` the caret offset
    in `before` (to tell two copies of the pasted text apart). Words are aligned with difflib; only replacements
    inside the pasted span count, each side 1-3 words that plausibly sound the same. Insertions, deletions,
    punctuation and plain case changes, edits outside the span and rewrites give nothing."""
    if not before or not pasted or before == after:
        return []
    caret_n = None if caret is None else len(_normalise(before[:max(0, caret)]))
    before, after, pasted = _normalise(before), _normalise(after or ""), _normalise(pasted).strip()
    span = locate(before, pasted, caret_n)
    if span is None:
        return []
    p0, p1 = span
    a = [(m.group(), m.start(), m.end()) for m in _TOKEN.finditer(before)]
    b = [m.group() for m in _TOKEN.finditer(after)]
    def in_span(t) -> bool:
        return t[1] < p1 and t[2] > p0

    inside = sum(1 for t in a if in_span(t))
    changed, candidates = 0, []
    matcher = difflib.SequenceMatcher(None, [t[0] for t in a], b, autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op in ("equal", "insert"):
            continue
        touched = sum(1 for t in a[i1:i2] if in_span(t))
        if not touched:
            continue
        changed += touched
        if op == "replace" and touched == i2 - i1:      # wholly inside the dictation
            candidates.append((i1, i2, j1, j2))
    if changed > REWRITE_WORDS and changed * 2 > inside:
        return []
    out: list[tuple[str, str]] = []
    for i1, i2, j1, j2 in candidates:
        # A few unchanged words either side go along, so a name keeps the words that make it one.
        lo = 0
        while lo < MAX_WORDS - 1 and i1 - lo > 0 and j1 - lo > 0 and a[i1 - lo - 1][0] == b[j1 - lo - 1] \
                and in_span(a[i1 - lo - 1]):
            lo += 1
        hi = 0
        while hi < MAX_WORDS - 1 and i2 + hi < len(a) and j2 + hi < len(b) and a[i2 + hi][0] == b[j2 + hi] \
                and in_span(a[i2 + hi]):
            hi += 1
        run = a[i1 - lo:i2 + hi]
        starts = [_sentence_start(before, t[1]) for t in run]
        for pair in _pairs_from([t[0] for t in run], b[j1 - lo:j2 + hi], starts):
            if pair not in out:
                out.append(pair)
    return out


# ---------------------------------------------------------------- lexicon

_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _path_lock(path: Path) -> threading.RLock:
    """One lock per lexicon file, shared by every Lexicon on it in this process."""
    key = os.path.normcase(os.path.abspath(path))
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.RLock())


def _decode(raw: bytes) -> str:
    """The file's text: UTF-16 when it opens with a UTF-16 byte order mark (Windows PowerShell 5.1's Out-File and
    > write that), else UTF-8 with or without a mark. Anything else raises UnicodeDecodeError, a ValueError."""
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig")


def _count(value) -> int:
    """A stored count, at least 1. One that is not a finite number (Infinity, NaN or 1e999 typed into the file)
    reads as 1 rather than setting the whole file aside."""
    if isinstance(value, float) and not math.isfinite(value):
        return 1
    return max(1, int(value))


def _parse(raw: str) -> dict[str, dict]:
    entries: dict[str, dict] = {}
    for e in json.loads(raw).get("entries", []):
        heard, written = _key(str(e["heard"])), _display(str(e["written"]))
        if heard and written:
            entries[heard] = {"heard": heard, "written": written, "count": _count(e.get("count", 1)),
                              "named": bool(e.get("named", False)),
                              "first": str(e.get("first", "")), "last": str(e.get("last", ""))}
    return entries


def _follows(written: str, named: bool) -> bool:
    """A named pair with one capital, its first: that capital was the dictation's (a sentence start, or a name
    Whisper knew), so apply() takes the capital from the text it respells."""
    return named and written[:1].isupper() and not _inner_capital(written)


def _protected(text: str, terms: Iterable[str]) -> list[tuple[int, int]]:
    """Where the given terms (the configured vocabulary) occur in the text: whole words, any case or spacing."""
    spans = []
    for t in terms or ():
        words = str(t).split()
        if words:
            pattern = r"(?<!\w)" + r"\s+".join(re.escape(w) for w in words) + r"(?!\w)"
            spans.extend(m.span() for m in re.finditer(pattern, text, re.IGNORECASE))
    return spans


class Lexicon:
    """Learned (heard, written) pairs in lexicon.json. Thread-safe, and safe for several Lexicons on one file in
    this process (they share a lock). Every change is written atomically, and the file is read again when
    something else changed it (the user may edit or delete it to reset). A file another process is replacing
    at that moment is read again shortly; only a file that is truly unreadable is set aside."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else LEXICON_PATH
        self._lock = _path_lock(self.path)
        self._entries: dict[str, dict] = {}
        self._stamp: tuple[int, int] | None = None
        self._version = 0
        self._compiled: tuple[int, re.Pattern | None, dict[str, tuple[str, bool]]] | None = None
        with self._lock:
            self._refresh()

    # -- file

    def _file_stamp(self) -> tuple[int, int] | None:
        try:
            st = self.path.stat()
        except OSError:
            return None
        return st.st_mtime_ns, st.st_size

    def _load(self, entries: dict[str, dict], stamp: tuple[int, int] | None) -> None:
        self._entries, self._stamp = entries, stamp
        self._version += 1

    def _refresh(self) -> None:
        stamp = self._file_stamp()
        if stamp == self._stamp:
            return
        if stamp is None:
            self._load({}, None)
            return
        for _ in range(_READ_TRIES):
            try:
                raw = self.path.read_bytes()
            except FileNotFoundError:
                self._load({}, None)
                return
            except OSError:                         # another process replacing it this instant
                time.sleep(_READ_WAIT_S)
                continue
            try:
                entries = _parse(_decode(raw))
            except (ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError) as exc:
                # ValueError covers bad JSON and a file in another encoding (UnicodeDecodeError).
                again = self._file_stamp()
                if again != stamp:                  # changed while it was read (an editor saving): read again
                    stamp = again
                    continue
                log.warning("lexicon unreadable, starting empty: %s", type(exc).__name__)
                try:
                    os.replace(self.path, self.path.with_name(self.path.name + ".bad"))
                except OSError:
                    pass
                self._load({}, self._file_stamp())
                return
            self._load(entries, stamp)
            return
        log.warning("lexicon busy; keeping the entries read before")   # the stamp stays old: read again next call

    def _save(self) -> None:
        self._version += 1
        entries = sorted(self._entries.values(), key=lambda e: e["heard"])
        data = json.dumps({"version": 1, "entries": entries}, ensure_ascii=False, indent=1)
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            for attempt in range(_REPLACE_TRIES):
                try:
                    os.replace(tmp, self.path)
                    break
                except PermissionError:               # a reader holds the file for a moment (sharing violation)
                    if attempt == _REPLACE_TRIES - 1:
                        raise
                    time.sleep(0.05)
        except OSError as exc:
            log.error("lexicon not saved: %r", exc)
            try:
                tmp.unlink()
            except OSError:
                pass
        self._stamp = self._file_stamp()

    # -- changes

    def add(self, heard: str, written: str) -> None:
        """Record one sighting of `written` where Scribe wrote `heard`, as Scribe wrote it (its capitals tell
        whether Whisper took it for a name). A correction back to a pair's heard side forgets that pair
        instead; a pair that changes a figure is not recorded (number words joined into a name are)."""
        heard_k, written_d = _key(heard or ""), _display(written or "")
        if not heard_k or not written_d or (heard or "").strip() == (written or "").strip():
            return
        if len(heard_k.split()) > MAX_WORDS or len(written_d.split()) > MAX_WORDS \
                or max(len(heard_k), len(written_d)) > MAX_PAIR_CHARS:
            return
        kept = _figures_kept(heard_k, written_d)
        if not kept and not _number_name(written_d, heard):
            return
        named = _named(heard, written)
        now = datetime.now().isoformat(timespec="seconds")
        with self._lock:
            self._refresh()
            undone = self._entries.get(_key(written_d)) if _key(written_d) != heard_k else None
            if undone is not None and _key(undone["written"]) == heard_k:
                del self._entries[undone["heard"]]
                log.info("forgot a pair")           # never the words: the log outlives a reset of the file
                self._save()
                return
            if not kept:
                return                              # a name written back as its number words only forgets it
            e = self._entries.get(heard_k)
            if e is not None and _key(e["written"]) == _key(written_d):
                e["count"] += 1
                e["last"] = now
                if e["named"] or not named:         # a sighting in small letters settles the written side's case
                    e["written"] = written_d
                e["named"] = e["named"] and named
            else:
                self._entries[heard_k] = {"heard": heard_k, "written": written_d, "count": 1, "named": named,
                                          "first": now, "last": now}
            if len(self._entries) > MAX_ENTRIES:
                ranked = sorted(self._entries.values(), key=lambda x: (x["count"], x["last"]))
                for old in ranked[:len(self._entries) - MAX_ENTRIES]:
                    del self._entries[old["heard"]]
            self._save()

    def remove(self, heard: str) -> None:
        with self._lock:
            self._refresh()
            if self._entries.pop(_key(heard or ""), None) is not None:
                self._save()

    # -- reading

    def __len__(self) -> int:
        with self._lock:
            self._refresh()
            return len(self._entries)

    def _ranked(self) -> list[dict]:
        """Entries most used first. One typed into the file that changes a figure is never used; number words
        joined into a name ("ten cent" -> "Tencent") are listed, and usable() keeps them out of apply()."""
        return sorted((e for e in self._entries.values() if _figures_kept(e["heard"], e["written"])),
                      key=lambda e: (e["count"], e["last"]), reverse=True)

    def pairs(self) -> list[tuple[str, str]]:
        """(heard, written) for every learned entry, most used first: hints for a step that sees the sentence.
        Unconfirmed pairs are among them, so they are not for respelling mechanically (see confirmed())."""
        with self._lock:
            self._refresh()
            return [(e["heard"], e["written"]) for e in self._ranked()]

    def confirmed(self) -> list[tuple[str, str]]:
        """The pairs apply() uses, most used first: the only ones safe to respell without reading the sentence."""
        with self._lock:
            self._refresh()
            return [(e["heard"], e["written"]) for e in self._ranked()
                    if self.usable(e["heard"], e["written"], e["count"], e["named"])]

    def terms(self) -> list[str]:
        """Written forms worth biasing Whisper towards (proper nouns, unusual spellings), most used first. For
        Whisper's prompt only: forcing their casing everywhere would be wrong ("Mark" is not "mark the book")."""
        with self._lock:
            self._refresh()
            out, seen = [], set()
            for e in self._ranked():
                w = e["written"]
                if is_special(w) and w.lower() not in seen:
                    seen.add(w.lower())
                    out.append(w)
                    if len(out) >= MAX_TERMS:
                        break
            return out

    @staticmethod
    def usable(heard: str, written: str, count: int, named: bool = False) -> bool:
        """The confirmation rule for apply() (module docstring)."""
        if not _same_figures(heard, written):
            return False                                # "ten cent" -> "Tencent" too: "it costs ten cent"
        if all(is_common(w) for w in heard.split()):
            return False
        if _core(heard) == _core(written):              # case or spacing only
            if not is_proper(written):
                return False
            if len(heard.split()) > 1 or len(written.split()) > 1 or _coined(written):
                return True
            return written.isupper() and count >= CONFIRM_COUNT
        if is_proper(written) and not named:
            return True
        return count >= CONFIRM_COUNT

    def _pattern(self) -> tuple[re.Pattern | None, dict[str, tuple[str, bool]]]:
        if self._compiled is None or self._compiled[0] != self._version:
            table = {e["heard"]: (e["written"], _follows(e["written"], e["named"])) for e in self._entries.values()
                     if self.usable(e["heard"], e["written"], e["count"], e["named"])}
            alts = [r"\s+".join(re.escape(w) for w in h.split()) for h in sorted(table, key=len, reverse=True)]
            rx = re.compile(r"(?<!\w)(?:" + "|".join(alts) + r")(?!\w)", re.IGNORECASE) if alts else None
            self._compiled = (self._version, rx, table)
        return self._compiled[1], self._compiled[2]

    def apply(self, text: str, protect: Iterable[str] = ()) -> str:
        """Respell confirmed pairs: whole words and phrases, any case, longest first, in one pass (no chains).
        Where Whisper wrote a capital (a sentence start, or a name) and the learned form starts lower case,
        the capital stays, unless the learned form is a coined spelling ("iPhone"). Text inside any `protect`
        term (pass the configured vocabulary, which asr.apply_vocabulary has already put right) is left alone."""
        if not text:
            return text
        with self._lock:
            self._refresh()
            rx, table = self._pattern()
        if rx is None:
            return text
        guard = _protected(text, protect)

        def swap(m: re.Match) -> str:
            found = m.group()
            if any(s < m.end() and m.start() < e for s, e in guard):
                return found
            written, follow = table.get(" ".join(found.lower().split()), (found, False))
            if follow:
                written = _lower_first(written)
            if found[:1].isupper() and written[:1].islower() and not _inner_capital(written):
                written = written[0].upper() + written[1:]
            return written

        return rx.sub(swap, text)


def _sentence_start(text: str, i: int) -> bool:
    while i > 0 and text[i - 1] in _OPENERS:
        i -= 1
    return i == 0 or text[i - 1] in _SENT_END


# ---------------------------------------------------------------- UI Automation

UIA_ValuePatternId = 10002
UIA_TextPatternId = 10014
_START, _END = 0, 1                    # TextPatternRangeEndpoint
_CHARACTER = 0                         # TextUnit_Character
_COINIT_MULTITHREADED = 0
_import_lock = threading.Lock()


def _comtypes():
    """comtypes and the UIAutomationClient wrapper. comtypes initialises COM on whichever thread imports it
    first; imported here first, that is the watcher thread, as a multithreaded apartment."""
    with _import_lock:
        if "comtypes" not in sys.modules:
            had, old = hasattr(sys, "coinit_flags"), getattr(sys, "coinit_flags", None)
            sys.coinit_flags = _COINIT_MULTITHREADED
            try:
                import comtypes  # noqa: F401
            finally:
                if had:
                    sys.coinit_flags = old
                else:
                    del sys.coinit_flags
        import comtypes
        import comtypes.client
        return comtypes, comtypes.client.GetModule("UIAutomationCore.dll")


class UIABackend:
    """UI Automation for EditWatcher, used only on the thread that called open()."""

    def __init__(self):
        self._uia = None
        self._mod = None
        self._com = None
        self._errors: tuple = (OSError,)

    def open(self) -> None:
        comtypes, mod = _comtypes()
        comtypes.CoInitializeEx(_COINIT_MULTITHREADED)
        self._com, self._mod = comtypes, mod
        self._errors = (comtypes.COMError, OSError, ValueError, AttributeError)
        from comtypes.client import CreateObject
        try:
            uia = CreateObject(mod.CUIAutomation8, interface=mod.IUIAutomation2)
            uia.ConnectionTimeout = UIA_TIMEOUT_MS
            uia.TransactionTimeout = UIA_TIMEOUT_MS
        except self._errors:
            uia = CreateObject(mod.CUIAutomation, interface=mod.IUIAutomation)
        self._uia = uia

    def close(self) -> None:
        self._uia = None
        gc.collect()                    # every COM pointer released before the apartment goes
        if self._com is not None:
            try:
                self._com.CoUninitialize()
            except OSError:
                pass
            self._com = None

    def focused(self):
        try:
            return self._uia.GetFocusedElement()
        except self._errors:
            return None

    def from_handle(self, hwnd: int):
        return self._uia.ElementFromHandle(hwnd)

    def same(self, a, b) -> bool:
        try:
            return bool(self._uia.CompareElements(a, b))
        except self._errors:
            return False

    def is_password(self, el) -> bool:
        try:
            return bool(el.CurrentIsPassword)
        except self._errors:
            return True                 # unreadable counts as a password box: never read

    def _pattern(self, el, pattern_id: int, iface):
        p = el.GetCurrentPattern(pattern_id)
        return p.QueryInterface(iface) if p else None

    def read(self, el, mode: str | None = None) -> tuple[str, int | None, str] | None:
        """(text, caret, mode). Mode "value" reads ValuePattern.CurrentValue, "text" a window of TextPattern
        around the caret; None picks one (value first) and is used for the snapshot. None when unreadable."""
        try:
            if mode in (None, "value"):
                vp = self._pattern(el, UIA_ValuePatternId, self._mod.IUIAutomationValuePattern)
                if vp is not None:
                    value = vp.CurrentValue or ""
                    if len(value) <= MAX_CHARS:
                        return value, (self._caret(el) if mode is None else None), "value"
                if mode == "value":
                    return None
            tp = self._pattern(el, UIA_TextPatternId, self._mod.IUIAutomationTextPattern)
            if tp is None:
                return None
            text, caret = self._window(tp)
            return text, caret, "text"
        except self._errors:
            return None

    def _selection(self, tp):
        sel = tp.GetSelection()
        if not sel or sel.Length < 1:
            return None
        caret = sel.GetElement(0).Clone()
        caret.MoveEndpointByRange(_END, caret, _START)          # collapse to the selection's start
        return caret

    def _caret(self, el) -> int | None:
        """Caret offset from the start of the box, through TextPattern, when the box has one."""
        try:
            tp = self._pattern(el, UIA_TextPatternId, self._mod.IUIAutomationTextPattern)
            caret = self._selection(tp) if tp is not None else None
            if caret is None:
                return None
            pre = tp.DocumentRange.Clone()
            pre.MoveEndpointByRange(_END, caret, _START)
            n = len(pre.GetText(MAX_CHARS + 1))
            return n if n <= MAX_CHARS else None
        except self._errors:
            return None

    def _window(self, tp) -> tuple[str, int | None]:
        """Up to MAX_CHARS of text centred on the caret, and the caret's offset in it."""
        caret = self._selection(tp)
        if caret is None:
            return tp.DocumentRange.GetText(MAX_CHARS), None
        half = MAX_CHARS // 2
        pre, post = caret.Clone(), caret.Clone()
        pre.MoveEndpointByUnit(_START, _CHARACTER, -half)
        post.MoveEndpointByUnit(_END, _CHARACTER, half)
        a, b = pre.GetText(half + 1), post.GetText(half + 1)
        return a + b, len(a)


# ---------------------------------------------------------------- watcher

class EditWatcher:
    """Watches the box a dictation was pasted into and teaches the Lexicon what the user corrected.

    One daemon thread with its own COM apartment. A watch ends, and is compared once, when the box changed and
    then held still for settle_s, focus moved to another element, the box was emptied (a chat box sending),
    max_s passed, or a new watch started. Only a reading seen on two polls in a row (or settled) counts, so a
    word caught half-typed is never learned.
    stop() drops the current watch unread. `backend` replaces UI Automation in tests."""

    def __init__(self, lexicon: Lexicon, *, poll_s: float = 1.5, max_s: float = 120.0, settle_s: float = 4.0,
                 backend=None):
        self.lexicon = lexicon
        self.poll_s, self.max_s, self.settle_s = poll_s, max_s, settle_s
        self._backend = backend
        self._cond = threading.Condition()
        self._request: tuple[str, int] | None = None
        self._seq = 0
        self._stopping = False
        self._broken = False
        self._thread: threading.Thread | None = None

    def watch(self, pasted: str) -> None:
        """Start watching the focused box for edits to `pasted`. Returns at once; ends any earlier watch."""
        if not pasted or not pasted.strip():
            return
        with self._cond:
            if self._broken:
                return
            self._seq += 1
            self._request = (pasted, self._seq)
            self._stopping = False
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="scribe-learn", daemon=True)
                self._thread.start()
            self._cond.notify_all()

    def stop(self) -> None:
        with self._cond:
            self._stopping = True
            self._request = None
            self._cond.notify_all()
            t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(STOP_JOIN_S)

    # -- thread

    def _run(self) -> None:
        backend = self._backend if self._backend is not None else UIABackend()
        try:
            backend.open()
        except Exception:
            log.exception("UI Automation unavailable; corrections are not learned this session")
            with self._cond:
                self._broken = True
            return
        try:
            while True:
                with self._cond:
                    self._cond.wait_for(lambda: self._stopping or self._request is not None)
                    if self._stopping:
                        return
                    pasted, seq = self._request
                    self._request = None
                try:
                    self._watch_one(backend, pasted, seq)
                except Exception:
                    log.exception("edit watch failed")
        finally:
            try:
                backend.close()
            except Exception:
                log.exception("UI Automation close failed")
            with self._cond:
                if self._thread is threading.current_thread():
                    self._thread = None
                    if self._request is not None and not self._stopping:   # a watch() raced this exit
                        self._thread = threading.Thread(target=self._run, name="scribe-learn", daemon=True)
                        self._thread.start()

    def _interrupted(self, seq: int) -> bool:
        return self._stopping or self._seq != seq

    def _sleep(self, seconds: float, seq: int) -> bool:
        """Wait; True when a new watch or stop() cut it short."""
        with self._cond:
            return self._cond.wait_for(lambda: self._interrupted(seq), timeout=seconds)

    def _snapshot(self, b, pasted: str, seq: int):
        """(element, (text, caret, mode)) once the pasted text shows in the focused box, else None."""
        want = _normalise(pasted).strip()
        deadline = time.monotonic() + FIND_WAIT_S
        while True:
            snap = None
            el = b.focused()
            if el is not None:
                if b.is_password(el):
                    log.info("edit watch skipped: password box")
                    return None
                snap = b.read(el, None)
                if snap is not None and want in _normalise(snap[0]):
                    return el, snap
            if time.monotonic() >= deadline or self._sleep(FIND_POLL_S, seq):
                log.info("edit watch skipped: %s", "no readable box" if el is None or snap is None
                         else "dictation not seen in the box")
                return None

    def _watch_one(self, b, pasted: str, seq: int) -> None:
        found = self._snapshot(b, pasted, seq)
        if found is None:
            return
        el, (before, caret, mode) = found
        found = None
        last, stable, changed_at = before, None, None
        start = time.monotonic()
        reason = None
        while reason is None:
            if self._sleep(self.poll_s, seq):
                if self._stopping:
                    return                                  # stop(): dropped unread
                reason = "new watch"
            now = time.monotonic()
            cur = b.read(el, mode)
            if cur is None:
                reason = reason or "box gone"
            elif not any(c.isalnum() for c in cur[0]):
                reason = reason or "sent"           # sent or cleared in place (a chat box): the steady reading stands
            elif cur[0] == last:
                if last != before:
                    stable = last
            else:
                last, changed_at = cur[0], now
            if reason:
                break
            if changed_at is not None and last != before and now - changed_at >= self.settle_s:
                reason, stable = "settled", last
            elif now - start >= self.max_s:
                reason = "time"
            else:
                f = b.focused()
                if f is not None and not b.same(f, el):
                    reason = "focus moved"
        el = cur = last = None
        if stable is None:
            log.debug("edit watch ended (%s): no edit", reason)
            return
        pairs = learn_pairs(before, stable, pasted, caret=caret)
        before = stable = None
        for heard, written in pairs:
            self.lexicon.add(heard, written)
        log.info("edit watch ended (%s): %d learned", reason, len(pairs))     # a count, never the words
