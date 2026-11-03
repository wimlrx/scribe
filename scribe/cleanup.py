"""Wispr Flow-style clean-up of a dictation by a local model (Ollama), between Whisper and the paste.

The model drops fillers, stutters, repeats and spoken self-corrections, punctuates, writes numbers as
digits, turns an announced list into a list, and spells the user's names and terms their way, including
names the speaker has corrected once (learned pairs, see relevant_pairs). It never answers, translates, adds content,
changes a figure or drops a negation: code guards check every output, and anything doubtful pastes Whisper's
text unchanged. Runs only against the local Ollama server, never through a proxy; no dictated text is logged
or kept.
"""
from __future__ import annotations

import bisect
import difflib
import logging
import re
import threading
import time
from collections import Counter
from typing import Callable

import requests

from . import learn

log = logging.getLogger("scribe.cleanup")

MIN_WORDS = 4                 # shorter dictations paste as Whisper wrote them
NUM_CTX = 8192                # the same in warm() and clean(): a different value would reload the model
CONNECT_S = 0.5               # the server is local; a slow connect means it is down
WARM_TIMEOUT = (3, 180)       # a cold load from disk
PREDICT_PER_WORD = 3          # num_predict bound: tokens per input word, plus a base
PREDICT_BASE = 64
PREDICT_MAX = 2048
MAX_PAIRS = 40                # learned pairs put in one prompt (only those heard in this dictation)
REPEAT_MIN_WORDS = 4          # a run this long said twice in a row is collapsed before the model sees it
SPELL_RATIO = 0.8             # a sound-alike word must also be spelt this close to a term's word to be swapped
TERM_RATIO = 0.7              # ... and this close for the model's own swap to pass the term guard
NO_PROXY = {"http": None, "https": None, "all": None}    # the server is local: no system or HTTP(S)/ALL_PROXY
                                                         # (a copy per call: requests adds to the dict it gets)

# Guards. Word counts use letter-bearing words only, so list numbers and digits do not count.
ADDED_MAX = 1                 # new words (or extra uses of the speaker's words) tolerated outright ...
ADDED_FRAC = 0.1              # ... or this share of the output, whichever is larger
LEN_LOW, LEN_HIGH, LEN_SLACK = 0.25, 1.3, 4
OPEN_SKIP = 3                 # words a dictation may say before the "Here is" or "Sure" the output opens with
AF_MIN_DISTINCT = 3           # the translation guard engages with this many distinct Afrikaans words
AF_MISSING_MAX = 0.6          # share of them allowed to vanish (self-corrections drop some)
AF_SHARE_MIN = 0.4            # output's Afrikaans share must keep this fraction of the input's
AF_NEW_MAX = 1                # this many Afrikaans words the dictation lacked means English was translated
JOIN_MAX = 12                 # spoken numbers read together as one figure: 'one two three four' -> 1234
NEG_WINDOW = 12               # a correction marker this many words after a negation may take it away

_WORD = re.compile(r"[^\W_]+")
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_TAG = re.compile(r"</?dictation>", re.I)
_SENTENCE = re.compile(r"(?<=[.?!])\s+|\n+")
_FENCE = re.compile(r"```[a-zA-Z]*\n?(.*?)\n?```", re.S)
_PREFACE = re.compile(
    r"^(?:here(?:'s|\s+is|\s+are)\b|sure\b|certainly\b|of\s+course\b|absolutely\b|cleaned\b|corrected\b|"
    r"the\s+(?:cleaned|corrected|tidied)\b|i(?:'ve|\s+have)\s+(?:cleaned|corrected|tidied)\b|below\s+is\b|"
    r"ok(?:ay)?,?\s+here\b|output\s*:|transcript\s*:)", re.I)

# Words the model may add without counting as new content: articles and joins for punctuation fixes,
# contraction and ordinal pieces (I'll, 6th), and the OK spelling.
FREE = frozenset({
    "a", "an", "the", "and", "or", "but", "so", "to", "of", "for", "in", "on", "at", "is", "are", "it",
    "i", "that", "this", "with", "as", "ok", "okay", "s", "t", "ll", "re", "ve", "d", "m", "n",
    "st", "nd", "rd", "th",
})

# Common Afrikaans words that are not also common English words; used only to spot a translation.
AFRIKAANS = frozenset("""
    ek jy jou ons hulle julle hy sy haar hom die nie het en van vir wat dit dat sal moet kan maar baie ook
    nou gaan na om te toe daar hier ja asseblief dankie goed reg dis nog sommer eers weer uit oor saam
    alles niks iets iemand waar hoekom hoe wanneer môre vandag gister skryf stuur sê doen maak kyk
    dink weet wil moes sou kon gesê gedoen gemaak gestuur vergadering maandag dinsdag woensdag donderdag
    vrydag saterdag sondag lekker mos darem seker dalk regtig eintlik omdat terwyl sodat voordat
    nadat hierdie daardie nommers elke almal geen nooit altyd soms gou groot klein nuwe mense geld maand jaar
    dag tyd beskikbaar miskien presies verstaan soos sonder tussen agter voor langs binne buite deur
    kry lê staan ry praat vra antwoord onthou vergeet klaar kliënt kliënte ouens ou nè hê
    asb volgende eerste tweede derde punte dinge laat stuk epos dan
""".split())
CORRECTION_WORDS = frozenset({"nee", "wag", "bedoel"})
FILLERS = frozenset({"um", "umm", "uh", "uhh", "uhm", "er", "erm", "ah", "eh", "hmm", "mm", "mhm"})
_SAME = {"s": "is", "okay": "ok"}     # "Here's" opens like "here is", "OK" like "okay"

# Afrikaans words and the English twins a translation puts in their place (a / between choices), for the
# code-switch of one or two words the share test is too coarse to see: "die meeting was goed" -> "The meeting
# was good", "dis fine" -> "It's fine".
TWINS = {a: tuple(e.split("/")) for a, e in (p.split("=") for p in """
    die=the en=and dit=it/this dis=this/it vir=for/to van=of/from met=with maar=but goed=good nie=not
    ek=i jy=you ons=we/us hy=he sy=she/his hulle=they/them het=have/has/had wat=what/which/that dat=that
    kan=can sal=will/shall moet=must/should wil=want gaan=go/going stuur=send skryf=write doen=do maak=make
    kyk=look dink=think weet=know praat=talk/speak vra=ask bel=call onthou=remember vergeet=forget
    verstaan=understand vandag=today môre=tomorrow gister=yesterday nou=now weer=again ook=also/too
    baie=very/many/much hier=here daar=there reg=right seker=sure dalk=maybe/perhaps miskien=maybe/perhaps
    regtig=really eintlik=actually altyd=always nooit=never soms=sometimes niks=nothing iets=something
    iemand=someone/somebody alles=everything almal=everyone/everybody waar=where hoekom=why hoe=how
    wanneer=when nuwe=new groot=big klein=small mense=people geld=money tyd=time dag=day maand=month
    jaar=year vergadering=meeting kliënt=client kliënte=clients nommers=numbers punte=points dinge=things
    volgende=next asseblief=please dankie=thanks/thank beskikbaar=available klaar=done/finished/ready
    saam=together sonder=without voor=before hierdie=this/these daardie=that/those ja=yes/yeah maandag=monday
    dinsdag=tuesday woensdag=wednesday
    donderdag=thursday vrydag=friday saterdag=saturday sondag=sunday
""".split())}

# Words that negate a clause, after expand() ("n't" is "not", "cannot" is "can not"), plus the spellings
# without an apostrophe, without/sonder ("without the sign-off" -> "with the sign-off") and neither/nor.
# Afrikaans "nie" and "moenie" (moet nie, which opens a "nie" pair) are counted on their own (double negation),
# "no" only as a determiner.
NEGATORS = frozenset("""
    not never nothing none nobody geen nooit niks niemand dont doesnt didnt cant wont isnt arent wasnt werent
    wouldnt couldnt shouldnt havent hasnt hadnt without sonder neither nor
""".split())
_NO_AFTER = frozenset("""
    i you we he she it they me him her us them the a an this that these those my your our his their there wait
    sorry no not never yes yeah ok okay so but and or then ja nee ek jy ons hy sy hulle dit die n
""".split())   # "no" before these is an interjection ("no we should", "no wait"), not "no reason"
# A "nor" followed by one of these opens a clause with its own negation ("I don't like it, nor does Adam"), so it is
# never part of the negation before it.
_AUX = frozenset("am is are was were do does did have has had will would shall should can could may might must".split())
_STOP = re.compile(r"[.?!\n]")     # a sentence ends
CORRECTIONS = ("no wait", "sorry", "i mean", "actually", "scratch that", "nee", "wag", "ek bedoel")
_PLAIN_MARKS = frozenset({"actually", "sorry"})     # also ordinary words: "not actually ready", "not sorry"
_CORRECTION = re.compile(r"(?<!\w)(?:" + "|".join(r"\W+".join(map(re.escape, c.split())) for c in CORRECTIONS)
                         + r")(?!\w)", re.I)

# Answers: never one of the words the output may add (ADDED_MAX) when the speaker did not say it.
ANSWERS = frozenset("yes yeah yep yup no nope sure correct ok okay ja nee".split())
# A lead-in ending ":" over a list line, the prompt's list format: a request the speaker asked as a question keeps its
# question there ("Can you check the following for me:\n- The pack"; see _list_questions). Questions are judged
# by "?" only.
_LIST_LEAD = re.compile(r"(?m):[ \t]*\n[ \t]*(?:\d+[.)]|[-•*])\s")

# Contractions written out for the guards, so "don't" is "do not", not the new word "don". 's and 'd stay.
_CONTRACTIONS = tuple((re.compile(p, re.I), r) for p, r in (
    (r"\b(w)on['’]t\b", r"\1ill not"), (r"\b(c)an['’]t\b", r"\1an not"), (r"\b(c)annot\b", r"\1an not"),
    (r"\b(sh)an['’]t\b", r"\1all not"), (r"n['’]t\b", " not"), (r"['’]ll\b", " will"), (r"['’]re\b", " are"),
    (r"['’]ve\b", " have"), (r"\b(i)['’]m\b", r"\1 am"),
))

# Values of spoken numbers, English and Afrikaans, for the figures guard. Ordinals count as their number (sixth
# -> 6th). Which words are numbers is learn.NUMBER_WORDS; a number word with no value here ('millions',
# 'fortieth') is compared as the word itself.
_ONES = {w: i for i, w in enumerate("zero one two three four five six seven eight nine".split())}
_ONES.update({w: i for i, w in enumerate("nul een twee drie vier vyf ses sewe agt nege".split())})
_TEENS = {w: i + 10 for i, w in enumerate("ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen "
                                          "nineteen".split())}
_TEENS.update({w: i + 10 for i, w in enumerate("tien elf twaalf dertien veertien vyftien sestien sewentien agtien "
                                               "negentien".split())})
_TENS = {w: (i + 2) * 10 for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
_TENS.update({w: (i + 2) * 10 for i, w in enumerate("twintig dertig veertig vyftig sestig sewentig tagtig "
                                                    "negentig".split())})
_ORD_ONES = {w: i + 1 for i, w in enumerate("first second third fourth fifth sixth seventh eighth ninth".split())}
_ORD_ONES.update({w: i + 1 for i, w in enumerate("eerste tweede derde vierde vyfde sesde sewende agtste "
                                                 "negende".split())})
_ORD_TEENS = {w: i + 10 for i, w in enumerate("tenth eleventh twelfth thirteenth fourteenth fifteenth sixteenth "
                                              "seventeenth eighteenth nineteenth".split())}
_ORD_TEENS["tiende"] = 10
_ORD_TENS = {"twentieth": 20, "thirtieth": 30}
_HUNDRED = frozenset({"hundred", "honderd"})
_SCALES = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12,
           "duisend": 1e3, "miljoen": 1e6, "miljard": 1e9}
_SUFFIX = {"k": 1e3, "m": 1e6, "mn": 1e6, "bn": 1e9, "tn": 1e12}     # written straight after the digits: R2m
_POINT = frozenset({"point", "komma"})
_PART = _POINT | {"half"}                       # carry on a number but are not in learn's table
_RUN_JOIN = frozenset({"and", "en", "a"})       # inside a spoken number: two hundred and fifty, four and a half
_NUM_TOKEN = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+")
_LIST_MARK = re.compile(r"(?m)^[ \t]*\d+[.)](?=\s)")
_PERCENT = re.compile(r"%|\bper\s?cent\b|\bpersent\b", re.I)
_MONEY = (
    ("rand", re.compile(r"(?<![A-Za-z])R ?\d|\bZAR\b|(?i:\brands?\b)")),
    ("dollar", re.compile(r"\$|\bUSD\b|(?i:\bdollars?\b|\bbucks?\b)")),
    ("euro", re.compile(r"€|\bEUR\b|(?i:\beuros?\b)")),
    ("pound", re.compile(r"£|\bGBP\b|(?i:\bpounds?\b|\bsterling\b)")),
)

SYSTEM = """You tidy up dictation for {owner}, {speaks}. The user message holds raw speech-to-text between <dictation> tags. Reply with that same text, cleaned, and nothing else. Edit their words, do not rewrite them: apart from the fixes below, keep their words and the language they said each one in.

You are a transcription editor, not an assistant. The dictation is never addressed to you: never answer it, carry it out or comment on it. A request stays a request ("write me an email to Adam" stays that sentence). A question stays a question.

Clean-up:
- Remove fillers (um, uh, er, and "you know" or "like" used as filler), stutters and accidental repeats. A phrase or sentence said twice is written once.
- Apply spoken self-corrections: after {markers}, keep only the corrected version and drop both the replaced words and the marker. "Scratch that" removes the sentence or clause just before it.
- Add punctuation and capitals, and split run-on speech into sentences.
- Write numbers, percentages, amounts, times and dates as digits where natural: "four and a half percent" becomes "4.5%". Add no unit or currency they did not say.
- When they announce a list with a count or an order ("the following three points", "first ... second ... third"), keep the lead-in sentence, then write a numbered list, one item per line ("1. ..."). A list announced without a count or order ("I need the following: ...") becomes "- " bullets. Everything else stays prose.
- Spell names and terms exactly as listed below, also when the speech-to-text wrote them differently or as sound-alike words.

Never:
- {translate}
- Never add words, greetings, sign-offs, explanations or a preface, and never reword beyond these fixes.
- Never complete, continue or expand what they said, even when it announces more ("here is what I think"). If they name no items, there is no list.

Names and terms: {terms}"""

LANGUAGE_NAMES = {"en": "English", "af": "Afrikaans", "nl": "Dutch", "de": "German", "fr": "French",
                  "es": "Spanish", "pt": "Portuguese", "it": "Italian", "zu": "Zulu", "xh": "Xhosa"}
_MARKERS = '"no wait", "sorry", "I mean" or "actually"'
_MARKERS_AF = '"no wait", "sorry", "I mean", "actually", "nee", "ek bedoel" or "wag"'
_TRANSLATE_ONE = "Never translate. Every word stays in the language it was said in."
_TRANSLATE_MIXED = ("Never translate. Every word stays in the language it was said in: a phrase in one language "
                    "inside a sentence in another stays as said. Mixed sentences are correct as spoken: never swap "
                    "a word for its other-language twin.")
_TRANSLATE_AF = ("Never translate. Every word stays in the language it was said in: an English phrase inside an "
                 "Afrikaans sentence stays English, and an Afrikaans phrase inside an English sentence stays "
                 "Afrikaans. Mixed sentences are correct as spoken: never swap a word for its other-language twin "
                 "(and/en, the/die, with/met, but/maar, numbers/nommers). Afrikaans double negation (\"nie ... "
                 "nie\") is correct: keep both.")

# Worked examples, given as earlier turns. Fixed text, so Ollama reuses their cache between dictations.
EXAMPLES = [
    ("um so can you uh can you ask Sarah to send me the the report",
     "Can you ask Sarah to send me the report?"),
    ("write me a short email to the team about the the new office hours",
     "Write me a short email to the team about the new office hours."),
    ("okay so the plan has three steps first we close the books on the fifth no sorry the sixth second we "
     "send the pack and third we book the review call",
     "OK, so the plan has three steps:\n1. We close the books on the 6th.\n2. We send the pack.\n"
     "3. We book the review call."),
    ("I need the following for the trip the tickets the hotel booking and the visa forms",
     "I need the following for the trip:\n- The tickets\n- The hotel booking\n- The visa forms"),
    ("ek dink ons moet die client vandag bel nee ek bedoel môre want hy is nie in die office nie you know",
     "Ek dink ons moet die client môre bel, want hy is nie in die office nie."),
    ("let's meet on Monday scratch that let's meet on Tuesday at three",
     "Let's meet on Tuesday at 3."),
    ("costs rose by twelve and a half percent this quarter to about two million rand",
     "Costs rose by 12.5% this quarter to about R2 million."),
    ("die meeting was goed but I think we need to send the minutes to everyone by Friday",
     "Die meeting was goed, but I think we need to send the minutes to everyone by Friday."),
    ("ja nee ons moet die kontrak eers teken and then we can start with the onboarding",
     "Ja nee, ons moet die kontrak eers teken, and then we can start with the onboarding."),
    ("so here are my thoughts on the budget for next year",
     "So here are my thoughts on the budget for next year."),
    ([("Stellen Bosch", "Stellenbosch")],
     "we drove out to stellen bosch for the um the lecture and it was it was good",
     "We drove out to Stellenbosch for the lecture, and it was good."),
]


# ---------------------------------------------------------------- prompt

def words(text: str) -> list[str]:
    """Lower-case words (letters and digits; apostrophes and hyphens split)."""
    return [w.lower() for w in _WORD.findall(text)]


def _alpha(tokens: list[str]) -> list[str]:
    return [w for w in tokens if any(c.isalpha() for c in w)]


def expand(text: str) -> str:
    """Contractions written out (don't -> do not, I'll -> I will), for comparing words; 's and 'd stay."""
    for rx, rep in _CONTRACTIONS:
        text = rx.sub(rep, text)
    return text


def _numeric(key: str) -> bool:
    return any(c.isdigit() for c in key) or bool(learn.number_words(key))


def collapse_repeats(text: str, min_words: int = REPEAT_MIN_WORDS) -> str:
    """Say-it-twice once: a run of `min_words` or more words repeated straight after itself keeps only its
    first copy ('I want it like X in the sense that I want it like X in the sense that I can' -> one copy).
    Words compare without case or punctuation. Shorter repeats, and any run with a number in it ('one two
    three four one two three four' is a code), are left to the model."""
    toks = text.split()
    keys = [" ".join(words(t)) for t in toks]
    changed = True
    while changed:
        changed = False
        for i in range(len(toks)):
            for n in range((len(toks) - i) // 2, min_words - 1, -1):
                if keys[i] == keys[i + n] and keys[i:i + n] == keys[i + n:i + 2 * n] and all(keys[i:i + n]) \
                        and not any(_numeric(k) for k in keys[i:i + n]):
                    del toks[i + n:i + 2 * n], keys[i + n:i + 2 * n]
                    changed = True
                    break
            if changed:
                break
    return " ".join(toks)


def _core(text: str) -> str:
    return "".join(c for c in text.lower() if c.isalnum())


def _phrase(text: str) -> str:
    """Its words as a whole-word pattern, any spacing between them."""
    return r"(?<!\w)" + r"\s+".join(re.escape(w) for w in text.split()) + r"(?!\w)"


def order_pair(heard: str, written: str) -> bool:
    """A learned pair the model may apply wherever it hears it ("write W where you hear H"). Not when the
    heard words are all everyday words ("with" -> "Alex" would hit every "with"), when one word only changed
    case ("mark" -> "Mark", "orca" -> "ORCA" depend on the sentence), or when one ordinary word replaced
    another ("cache" -> "cash"): those wait for learn.Lexicon to confirm them, and its apply() respells them
    before clean-up. Nor when a number word the speaker said goes into a name ("ten cent" -> "Tencent", "one drive" ->
    "OneDrive"): "it costs ten cent" is a figure, so such a pair stays a hint in learn.Lexicon. What is left is
    names and coined spellings (Kira, BlueSky, example.com, Q3)."""
    hw, ww = heard.split(), written.split()
    if not hw or not ww or all(learn.is_common(w) for w in hw):
        return False
    if learn.number_words(heard) != learn.number_words(written) and not re.search(r"\d", written):
        return False
    coined = bool(re.search(r"\d|[^\w\s'’-]", written)) or any(
        any(c.isupper() for c in w[1:]) and not w.isupper() for w in ww)
    if _core(heard) == _core(written) and len(hw) == len(ww) == 1 and not coined:
        return False
    return coined or learn.is_proper(written)


def relevant_pairs(text: str, pairs: list[tuple[str, str]], limit: int = MAX_PAIRS) -> list[tuple[str, str]]:
    """Learned (heard, written) pairs whose heard words occur in this dictation, as whole words in any case,
    and which may be applied as orders (order_pair). Pass learn.Lexicon.pairs, most used first: the first of
    a clash wins and the first `limit` are kept. Confirmed pairs are normally gone by now (Lexicon.apply runs
    first and respells them), so what reaches the model is the names the speaker has corrected only once."""
    found: dict[str, tuple[str, str]] = {}
    for pair in pairs or []:
        try:
            heard, written = (" ".join(str(p).split()) for p in pair)
        except (TypeError, ValueError):
            continue
        if not heard or not written or heard == written or not order_pair(heard, written):
            continue
        if heard.lower() not in found and re.search(_phrase(heard), text, re.I):
            found[heard.lower()] = (heard, written)
            if len(found) >= limit:
                break
    return list(found.values())


def _user_message(text: str, pairs: list[tuple[str, str]]) -> str:
    head = ""
    if pairs:
        head = "Spelling for this dictation: " + "; ".join(
            f'write "{w}" where you hear "{h}"' for h, w in pairs) + ".\n"
    return f"{head}<dictation>\n{text}\n</dictation>"


def clean_terms(terms: list[str]) -> list[str]:
    """Distinct non-empty terms, single-spaced, first spelling kept."""
    seen: dict[str, str] = {}
    for t in terms or []:
        if t is None:
            continue
        t = " ".join(str(t).split())
        if t and t.lower() not in seen:
            seen[t.lower()] = t
    return list(seen.values())


def _languages(languages) -> list[str]:
    """Configured language codes, lower-case, distinct, in order; English when none are set."""
    codes: list[str] = []
    for code in languages if isinstance(languages, (list, tuple)) else [languages] if languages else []:
        code = str(code or "").strip().lower()
        if code and code not in codes:
            codes.append(code)
    return codes or ["en"]


def system_prompt(terms: list[str], owner_name: str = "", languages=None) -> str:
    """SYSTEM filled in: the owner's first name ("the user" when none is set), the configured languages,
    and the terms. The Afrikaans markers and twins appear only when "af" is configured."""
    terms = clean_terms(terms)
    first = (str(owner_name or "").split() or [""])[0]
    codes = _languages(languages)
    names = [LANGUAGE_NAMES.get(c, c) for c in codes]
    spoken = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    if len(names) > 1:
        speaks = f"who speaks {spoken} and often switches language mid-sentence"
        translate = _TRANSLATE_AF if "af" in codes else _TRANSLATE_MIXED
    else:
        speaks = f"who speaks {spoken}"
        translate = _TRANSLATE_AF if "af" in codes else _TRANSLATE_ONE
    return SYSTEM.format(owner=first or "the user", speaks=speaks, translate=translate,
                         markers=_MARKERS_AF if "af" in codes else _MARKERS,
                         terms=", ".join(terms) + "." if terms else "none.")


def build_messages(text: str, terms: list[str], pairs: list[tuple[str, str]] | None = None,
                   owner_name: str = "", languages=None) -> list[dict]:
    """The chat for one dictation: rules and terms, the worked examples, then the dictation itself.
    `pairs` are the learned pairs that apply to this dictation (see relevant_pairs); `owner_name` and
    `languages` come from settings (see system_prompt)."""
    msgs = [{"role": "system", "content": system_prompt(terms, owner_name, languages)}]
    for ex in EXAMPLES:
        ex_pairs, said, clean = ex if len(ex) == 3 else ([], *ex)
        msgs.append({"role": "user", "content": _user_message(said, ex_pairs)})
        msgs.append({"role": "assistant", "content": clean})
    msgs.append({"role": "user", "content": _user_message(" ".join(text.split()), list(pairs or []))})
    return msgs


# ---------------------------------------------------------------- output checks

def tidy(raw: str, source: str) -> str:
    """The model's reply as pasteable text: no think block, echoed tags, code fence or wrapping quotes;
    LF line ends, no trailing spaces, at most one blank line in a row."""
    out = _THINK.sub("", raw or "").replace("\r\n", "\n").replace("\r", "\n")
    out = _TAG.sub("", out).strip()
    m = _FENCE.fullmatch(out)
    if m:
        out = m.group(1).strip()
    src = source.strip()
    if len(out) >= 2 and out[0] in "\"“" and out[-1] in "\"”" and not (src[:1] in "\"“" and src[-1:] in "\"”"):
        out = out[1:-1].strip()
    out = "\n".join(line.rstrip() for line in out.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def soundex(word: str) -> str:
    """American Soundex of a word's letters ('' for none): whisper and wispr are both W216."""
    letters = [c for c in word.lower() if "a" <= c <= "z"]
    if not letters:
        return ""
    codes = {**dict.fromkeys("bfpv", "1"), **dict.fromkeys("cgjkqsxz", "2"), **dict.fromkeys("dt", "3"),
             "l": "4", **dict.fromkeys("mn", "5"), "r": "6"}
    out, prev = letters[0].upper(), codes.get(letters[0], "")
    for c in letters[1:]:
        code = codes.get(c, "")
        if code and code != prev:
            out += code
        if c not in "hw":
            prev = code
    return (out + "000")[:4]


def _term_regex(term: str) -> str:
    """Whole-word, any case, any spacing or hyphen between its parts: BlueSky matches 'blue sky',
    Help Desk 'help-desk', and a one-word term of 6+ letters split once ('morning star')."""
    pieces: list[str] = []
    for w in term.split():
        pieces.extend(re.sub(r"(?<=[a-z])(?=[A-Z])", " ", w).split())
    sep = r"[\s-]*"
    if len(pieces) == 1 and len(pieces[0]) >= 6 and pieces[0].isalpha():
        p = pieces[0]
        alts = [re.escape(p)] + [re.escape(p[:i]) + r"[\s-]+" + re.escape(p[i:]) for i in range(2, len(p) - 1)]
        body = "(?:" + "|".join(alts) + ")"
    else:
        body = sep.join(re.escape(p) for p in pieces)
    return r"(?<!\w)" + body + r"(?!\w)"


def sounds_like(got: str, part: str, ratio: float = SPELL_RATIO) -> bool:
    """`got` is a misspelling of a term's word, not another word: the same Soundex, spelt within `ratio`,
    and not the word with letters added or cut at an edge (desks, frontal). 'whisper' is 'Wispr' and
    'optimiser' is 'Optimizer'; 'is' is not 'IQ', 'optimisation' is not 'Optimizer'."""
    g, p = got.lower(), part.lower()
    if g == p:
        return True
    if soundex(g) != soundex(p) or g.startswith(p) or p.startswith(g) or g.endswith(p) or p.endswith(g):
        return False
    return difflib.SequenceMatcher(None, g, p, autojunk=False).ratio() >= ratio


def respell(text: str, terms: list[str], ratio: float = SPELL_RATIO) -> str:
    """Make sure the model's text uses the user's spellings: each term in any case or spacing, then a
    multi-word term with one word exact and the others misspellings of it ('Whisper Flow' -> 'Wispr Flow';
    see sounds_like). Only for the configured vocabulary: learned pairs and terms stay with the model,
    since forcing them without reading the sentence is wrong ('Mark' is not 'mark the book')."""
    for term in sorted(clean_terms(terms), key=len, reverse=True):
        text = re.sub(_term_regex(term), lambda m, t=term: t, text, flags=re.I)
        parts = term.split()
        if len(parts) < 2 or not all(p.isalpha() for p in parts):
            continue
        pattern = r"(?<!\w)" + r"\s+".join([r"([^\W\d_]+)"] * len(parts)) + r"(?!\w)"

        def swap(m, parts=parts, term=term):
            got = m.groups()
            if all(sounds_like(g, p, ratio) for g, p in zip(got, parts)) \
                    and any(g.lower() == p.lower() for g, p in zip(got, parts)):
                return term
            return m.group(0)
        text = _overlapping_sub(pattern, swap, text)
    return text


def _overlapping_sub(pattern: str, fn, text: str) -> str:
    """re.sub that tries every word start, so 'the whisper flow' still finds 'whisper flow'."""
    rx = re.compile(pattern)
    out, i = [], 0
    while i < len(text):
        m = rx.match(text, i) if (i == 0 or not text[i - 1].isalnum()) and text[i].isalpha() else None
        if m:
            rep = fn(m)
            if rep != m.group(0):
                out.append(rep)
                i = m.end()
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _spoken(ws: list[str]) -> list[float | str]:
    """A run of number words as the numbers it says: 'two thousand and twenty six' -> [2026], 'two point
    three billion' -> [2.3e9], 'four and a half' -> [4.5], 'vier en twintig' -> [24]. A word that cannot
    carry on the number before it starts a new one: 'twenty twenty six' -> [20, 26], 'one two' -> [1, 2].
    A number word with no value in the tables stands for itself: 'millions' -> ['millions']."""
    out: list[float | str] = []
    total, cur, kind = 0.0, None, ""

    def close():
        nonlocal total, cur, kind
        if cur is not None or total:
            out.append(total + (cur or 0))
        total, cur, kind = 0.0, None, ""

    i = 0
    while i < len(ws):
        w, nxt = ws[i], ws[i + 1] if i + 1 < len(ws) else ""
        if w in _POINT:
            j, digits = i + 1, ""
            while j < len(ws) and ws[j] in _ONES:
                digits, j = digits + str(_ONES[ws[j]]), j + 1
            if digits and (cur is not None or total):
                v = total + (cur or 0) + float("0." + digits)
                if j < len(ws) and ws[j] in _SCALES:
                    v, j = v * _SCALES[ws[j]], j + 1
                total, cur, kind = 0.0, None, ""
                out.append(v)
                i = j
                continue
            close()
        elif w in _RUN_JOIN:
            if w != "a" and cur is not None and nxt in ("a", "half"):
                half_at = i + 2 if nxt == "a" else i + 1
                if half_at < len(ws) and ws[half_at] == "half":
                    cur, kind, i = cur + 0.5, "half", half_at + 1
                    continue
            if w == "en" and kind == "one" and cur is not None and cur < 10 and nxt in _TENS:
                cur, kind, i = cur + _TENS[nxt], "ten", i + 2       # Afrikaans: vier en twintig
                continue
        elif w == "half":
            close()
            out.append(0.5)
        elif w in _HUNDRED:
            if kind not in ("one", "teen"):
                close()
            cur, kind = (cur or 1) * 100, "hundred"
        elif w in _SCALES:
            total, cur, kind = total + (1 if cur is None else cur) * _SCALES[w], None, "scale"
        else:
            v = _ONES.get(w, _ORD_ONES.get(w))
            if v is not None:
                if not (kind in ("hundred", "scale") or (kind == "ten" and cur is not None and cur % 10 == 0)):
                    close()
                cur, kind = (cur or 0) + v, "one"
            else:
                v = _TEENS.get(w, _ORD_TEENS.get(w, _TENS.get(w, _ORD_TENS.get(w))))
                if v is not None:
                    if kind not in ("hundred", "scale"):
                        close()
                    cur, kind = (cur or 0) + v, "ten" if v >= 20 else "teen"
                else:
                    close()
                    out.append(w)
        i += 1
    close()
    return out


def _digit_value(tok: str) -> float | None:
    """'25,000' -> 25000, '4.5' -> 4.5, '2,5' -> 2.5 (Afrikaans decimal comma); None when unclear."""
    if "," in tok and "." in tok:
        tok = tok.replace(",", "")
    elif "," in tok:
        parts = tok.split(",")
        if all(len(p) == 3 for p in parts[1:]):
            tok = tok.replace(",", "")
        elif len(parts) == 2:
            tok = tok.replace(",", ".")
        else:
            return None
    try:
        return float(tok)
    except ValueError:
        return None


def _figures(text: str) -> list[tuple[int, int, list[tuple], list[str]]]:
    """The figures in a text in reading order, as (start, end, atoms, words). A written figure is one atom,
    (value with its scale, digits): 'R2.3 billion' -> (2.3e9, '23'), 'R2m' -> (2e6, '2'). A run of number
    words has an atom per number it says (_spoken, digits None) and keeps its words; learn.number_words finds
    them, compounds written as one word included (tweehonderd, vyfentwintig). List markers at a line start
    and ':00' are not figures: they are blanked, so start and end are still places in `text`."""
    text = _LIST_MARK.sub(lambda m: " " * len(m.group()), re.sub(r"(?<=\d):00(?!\d)", "   ", text))
    toks = [(m.group().lower(), m.start(), m.end()) for m in _NUM_TOKEN.finditer(text)]
    figs: list[tuple[int, int, list[tuple], list[str]]] = []
    run: list[str] = []
    span = [0, 0]

    def flush():
        while run and run[-1] in _RUN_JOIN:
            run.pop()
        if any(w not in _RUN_JOIN and w not in _POINT for w in run):
            figs.append((span[0], span[1], [(v, None) for v in _spoken(run)], list(run)))
        run.clear()

    i = 0
    while i < len(toks):
        t, start, end = toks[i]
        if t[0].isdigit():
            flush()
            v = _digit_value(t)
            if v is not None and i + 1 < len(toks):
                nt, ns, ne = toks[i + 1]
                if ns == end and nt in _SUFFIX:
                    v, i, end = v * _SUFFIX[nt], i + 1, ne
                elif nt in _SCALES:
                    v, i, end = v * _SCALES[nt], i + 1, ne
            figs.append((start, end, [(v, re.sub(r"\D", "", t))], []))
        elif (parts := learn.number_words(t) or ([t] if t in _PART else [])) or (run and t in _RUN_JOIN):
            if not run:
                span[0] = start
            span[1] = end
            for k, p in enumerate(parts or [t]):
                if k and parts[k - 1] in _ONES and p in _TENS:
                    run.append("en")                   # vyfentwintig is five-and-twenty
                run.append(p)
        else:
            flush()
        i += 1
    flush()
    return figs


def _same(a, b) -> bool:
    """Two figures are one: numbers within rounding, or the same number word."""
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return a is not None and b is not None and abs(a - b) <= 1e-9 * max(1.0, abs(b))


def _joinable(v) -> bool:
    return isinstance(v, float) and v == int(v) and 0 <= v < 1000


def _money(text: str) -> set[str]:
    return {name for name, rx in _MONEY if rx.search(text)}


def figures_ok(source: str, output: str) -> bool:
    """The output's figures are figures the speaker said, in the order said, each used once: no new, changed,
    swapped or repeated number, and no % or currency the speaker did not say. Spoken figures may come back as digits
    ('one point two five percent' -> '1.25%'), and spoken numbers in a row as one ('twenty twenty six' ->
    '2026', 'one two three four' -> '1234'). A figure may be left out (a self-correction), but not the one
    after a correction marker while the one before it stays ('twelve, I mean fifteen' -> '12'). An ordinal
    word stands for the adverb the speaker said ('firstly' -> 'First,', 'eerstens' -> 'Eerste')."""
    said = [(v, d, n, s, e) for n, (s, e, atoms, _) in enumerate(_figures(source)) for v, d in atoms]
    used = [False] * len(said)

    def fits(v, d, j: int, e: int) -> bool:
        """The output figure (v, d) is said[j:e]: one figure, or spoken numbers 0-999 in a row read as one."""
        if e - j == 1:
            sv, sd = said[j][:2]
            if v is None:                       # digits whose value is unclear ('1,2,3')
                return d == sd or (bool(d) and _same(float(d), sv))
            return _same(v, sv)
        block = said[j:e]
        if any(b[2] != block[0][2] or b[1] is not None or not _joinable(b[0]) for b in block):
            return False
        joined = "".join(str(int(b[0])) for b in block)
        return d == joined if v is None else _same(v, float(joined))

    def take(v, d, p: int) -> tuple[int, int] | None:
        """Tthe speaker said figures from p on that the output figure is, the stretch that ends first."""
        for e in range(p + 1, len(said) + 1):
            for j in range(e - 1, max(p, e - JOIN_MAX) - 1, -1):
                if fits(v, d, j, e):
                    return j, e
        return None

    adverbs = Counter(words(source))
    p = 0
    for _, _, atoms, run in _figures(output):
        forms = [next((f for f in (w + "ly", w + "ns") if adverbs[f] > 0), None) for w in run if w not in _RUN_JOIN]
        if run and all(forms):
            adverbs.subtract(forms)
            continue
        plans, spans, q = [], [], p
        for v, d in atoms:
            got = take(v, d, q)
            if got is None:
                break
            spans.append(got)
            q = got[1]
        else:
            plans.append(spans)
        if run and len(atoms) > 1 and all(_joinable(v) for v, _ in atoms):
            got = take(float("".join(str(int(v)) for v, _ in atoms)), None, p)
            if got:
                plans.append([got])
        if not plans:
            return False
        spans = min(plans, key=lambda s: s[-1][1])
        for j, e in spans:
            used[j:e] = [True] * (e - j)
        p = spans[-1][1]
    marks = [m.start() for m in _CORRECTION.finditer(source)]
    for (xv, _, xn, _, xe), (yv, _, yn, ys, _), xu, yu in zip(said, said[1:], used, used[1:]):
        if xu and not yu and xn != yn and not _same(xv, yv) and any(xe <= m < ys for m in marks):
            return False                        # the corrected figure kept, the correction dropped
    if len(_PERCENT.findall(output)) > len(_PERCENT.findall(source)):
        return False
    return not (_money(output) - _money(source))


def terms_ok(source: str, output: str, terms: list[str]) -> bool:
    """No multi-word term appears more often than the dictation said it, exactly or misspelt (respell's
    rule at TERM_RATIO): the model may write 'Wispr Flow' for 'whisper flow', never 'Project Planner'
    for 'project planning' or 'Capital IQ' for 'capital is'. One-word terms are names, left to it."""
    for term in clean_terms(terms):
        if len(term.split()) < 2:
            continue
        rx = re.compile(_term_regex(term), re.I)
        if len(rx.findall(output)) > len(rx.findall(respell(source, [term], TERM_RATIO))):
            return False
    return True


def _unstutter(toks: list[tuple]) -> list[tuple]:
    """(word, negates, ...) tokens, fillers already out, with a word or a phrase of any length said twice in a
    row kept once ('not not', 'we can not um we can not', or a run collapse_repeats gave the model once): the
    model drops such repeats, and no negation goes with them. A word of the copy kept negates if it did in
    either; the rest of its token is the first copy's."""
    out: list[tuple] = []
    ws: list[str] = []
    for tok in toks:
        out.append(tok)
        ws.append(tok[0])
        for k in range(1, len(ws) // 2 + 1):
            if ws[-1] == ws[-1 - k] and ws[-k:] == ws[-2 * k:-k]:
                out[-2 * k:-k] = [(a[0], a[1] or b[1], *a[2:]) for a, b in zip(out[-2 * k:-k], out[-k:])]
                del out[-k:], ws[-k:]
                break
    return out


def _negations(text: str, punct: bool = True) -> tuple[list[int], list[int], list[tuple[int, str]], list[int],
                                                       list[str]]:
    """Where a text negates, as word indexes once expand() and _unstutter have run: the negators, the "nie"s
    and "moenie"s, the correction markers (index, first word), the tied "nor"s, and the words. "no" counts only
    as a determiner: straight before a word ("no reason"), fillers skipped, not before a pronoun ("no wait"),
    and, when `punct`, not before punctuation ("No, Friday works"). Straight before "thanks" or "thank" it
    always counts, whatever punctuation lies between, on both texts: "got no thanks for it" negates, a decline
    ("No thanks, I will ...") keeps its "no", and "No, thanks" is the speaker's "no thanks". A "nor" is tied when a
    negator comes before it in its sentence and it opens no clause of its own (_AUX): "not the deck nor the
    pack" is one negation."""
    text = expand(text)
    toks = [(m.group().lower(), m.start(), m.end()) for m in _WORD.finditer(text)]
    toks = [t for t in toks if t[0] not in FILLERS]     # a filler splits no stutter and stands for no word
    det = [w == "no" and i + 1 < len(toks) and (toks[i + 1][0] in ("thanks", "thank") or toks[i + 1][0] not in
           _NO_AFTER and not (punct and re.search(r"[^\w\s]", text[end:toks[i + 1][1]])))
           for i, (w, _, end) in enumerate(toks)]
    stops = [m.start() for m in _STOP.finditer(text)]
    toks = _unstutter([(w, w in NEGATORS or d, bisect.bisect(stops, start))
                       for (w, start, _), d in zip(toks, det)])
    ws = [w for w, _, _ in toks]
    strict = [i for i, (_, neg, _) in enumerate(toks) if neg]
    nie = [i for i, w in enumerate(ws) if w in ("nie", "moenie")]
    marks = [(i, c.split()[0]) for i in range(len(ws)) for c in CORRECTIONS
             if ws[i:i + len(c.split())] == c.split()]
    tied = [i for k, i in enumerate(strict) if ws[i] == "nor" and k and toks[strict[k - 1]][2] == toks[i][2]
            and (i + 1 == len(ws) or ws[i + 1] not in _AUX)]
    return strict, nie, marks, tied, ws


def negation_ok(source: str, output: str) -> bool:
    """The output negates where the speaker did: as many of not, never, nothing, none, nobody, without, neither, nor, a
    determiner "no", geen, nooit, niks, niemand and sonder, and the "nie"s and "moenie"s the speaker said, give or take
    one: once the speaker said two, one may go (a pair falls to one), and one may close an odd one ('nie seker' -> 'nie
    seker nie'). A negation may go when a correction marker follows it within NEG_WINDOW words ('not happy with
    the fee, actually he is fine'), judged for each negation, not for the whole dictation. "actually" and
    "sorry" straight after the negation, or as the last word, are plain words ('not actually ready'). On a
    dictation with no punctuation, the model's comma after "no" ("No, Adam said") does not make it an
    interjection. A tied "nor" is part of the negation before it, so it may be written as an "or" ('did not
    send the deck nor the pack' -> 'did not send the deck or the pack'); without a new "or" it still counts
    ('nor' -> 'and'). An "or" written as a "nor" is a negation added ('we did not call and we sent the deck or
    the pack' -> '... nor the pack')."""
    punct = bool(re.search(r"[.,;:?!]", source))
    strict, nie, marks, tied, ws = _negations(source, punct)
    out_strict, out_nie, _, _, out_ws = _negations(output, punct)
    n = len(ws)

    def corrected(i: int) -> bool:
        return any(i < j <= i + NEG_WINDOW and not (w in _PLAIN_MARKS and j in (i + 1, n - 1)) for j, w in marks)

    nors, out_nors, ors, out_ors = ws.count("nor"), out_ws.count("nor"), ws.count("or"), out_ws.count("or")
    need = sum(not corrected(i) for i in strict)
    if out_ors > ors:                                # the speaker's tied "nor" written as an "or"
        need -= min(sum(not corrected(i) for i in tied), max(0, nors - out_nors))
    need_nie = sum(not corrected(i) for i in nie)
    low = need_nie - 1 if need_nie >= 2 else need_nie
    return need <= len(out_strict) <= len(strict) and low <= len(out_nie) <= len(nie) + len(nie) % 2


def _echoes(sentences: list[list[str]], budget: Counter) -> bool:
    """A sentence made only of the speaker's words that earlier sentences already used up: an answer restating them
    ('... should we use Postgres or SQLite? Use SQLite.')."""
    left = budget.copy()
    for content in sentences:
        if content and not any(left[w] > 0 for w in content):
            return True
        left.subtract(content)
    return False


def _list_questions(source: str, output: str, known: set[str] = frozenset()) -> int:
    """Lead-ins ending ":" over a list made of the words of a question the speaker asked, give or take ADDED_MAX, and
    at most half of its own words new ("Can you check the following for me:" for "Can you check these for me,
    the pack, ...?"): that question, kept as a list. `source` is the speaker's text respelt; a word of a name in `known`
    that sounds like one of the question's words is the speaker's ("kiera" -> "Kira")."""
    asked = [set(words(expand(q))) for q in re.findall(r"[^.?!]*\?", source)]
    n = 0
    for m in _LIST_LEAD.finditer(output):
        lead = {w for w in _alpha(words(expand(re.split(r"[.?!\n]", output[:m.start()])[-1]))) if w not in FREE}
        n += bool(lead) and any(len({w for w in lead - q if not (w in known and any(sounds_like(s, w) for s in q))})
                                <= min(ADDED_MAX, len(lead) // 2) for q in asked)
    return n


def _opens_like(src: list[str], out: list[str], n: int) -> bool:
    """The output's first n + 2 words are what the dictation opens with, after fillers and up to OPEN_SKIP
    words ('um here is the plan' may come back as 'Here is the plan'). n is the length of the preface match."""
    s = [_SAME.get(w, w) for w in src if w not in FILLERS]
    k = min(len(out), n + 2)
    head = [_SAME.get(w, w) for w in out[:k]]
    return bool(head) and any(s[i:i + k] == head for i in range(min(OPEN_SKIP, len(s)) + 1))


def _without_corrections(tokens: list[str]) -> list[str]:
    """Input words minus the Afrikaans self-correction markers (nee, wag, ek bedoel)."""
    out: list[str] = []
    for w in tokens:
        if w in CORRECTION_WORDS:
            if w == "bedoel" and out and out[-1] == "ek":
                out.pop()
            continue
        out.append(w)
    return out


def _kept_afrikaans(source: str) -> Counter:
    """The speaker's Afrikaans words the output is to keep: fillers out, a stutter once (_unstutter), and not the
    correction markers or a word one follows within NEG_WINDOW words (judged as negation_ok judges a negation)."""
    ws = [w for w, _ in _unstutter([(w, False) for w in words(expand(source)) if w not in FILLERS])]
    spans = [(i, i + len(c.split())) for i in range(len(ws)) for c in CORRECTIONS
             if ws[i:i + len(c.split())] == c.split()]
    return Counter(w for i, w in enumerate(ws) if w in AFRIKAANS and not any(a <= i < b for a, b in spans)
                   and not any(i < a <= i + NEG_WINDOW and not (ws[a] in _PLAIN_MARKS and a in (i + 1, len(ws) - 1))
                               for a, _ in spans))


def check(source: str, output: str, terms: list[str] | None = None,
          pairs: list[tuple[str, str]] | None = None) -> str | None:
    """The first guard the output fails, or None when it may be pasted. Words compare with contractions
    written out ("don't" is "do not").

    empty: nothing left. preface: opens with "Here is", "Sure", "Cleaned" ... where the dictation did not
    open that way. length: word count outside a sane band around the input's. figures: a number, % or
    currency the speaker did not say, or the speaker's figures swapped, repeated or changed (figures_ok).
    term: a multi-word term made out of ordinary words (terms_ok). translation: the dictation's Afrikaans is
    gone and other words came in, an Afrikaans word gave way to its English twin or back (TWINS; a stutter
    counted once), one of the speaker's Afrikaans words went while a word the speaker did not say came in
    ("lekker" -> "nice"; the speaker's words counted as the model is to keep them, _kept_afrikaans), or
    English turned Afrikaans. negation: a negation lost or added (negation_ok).
    added: more new words, or extra uses of the speaker's words, than a small allowance; an answer word the
    speaker did not say ("Yes, the terminal is working today."); fewer question marks than the speaker's (a
    ":" lead-in over a list made of one of the speaker's questions counts as one, _list_questions); or a
    sentence with none of the speaker's words left unused (the model answered: "Done.", "... SQLite? Use
    SQLite."); a word of a name from `terms` is worth one use when the speaker said a sound-alike of it
    ("kiera" -> "- Kira"), so a name the speaker did not say is still an answer ("Bloomberg.").
    dropped: none of the speaker's words are left ("It is.")."""
    if not output.strip():
        return "empty"
    src, out = words(expand(source)), words(expand(output))
    m = _PREFACE.match(output.lstrip(" \"'*#>-"))
    if m and not _opens_like(src, out, len(words(expand(m.group(0))))):
        return "preface"
    n_in, n_out = len(_alpha(src)), len(_alpha(out))
    if n_out > n_in * LEN_HIGH + LEN_SLACK or (n_out < n_in * LEN_LOW and n_in - n_out > LEN_SLACK):
        return "length"
    if not figures_ok(source, output):
        return "figures"
    if not terms_ok(source, output, terms or []):
        return "term"
    known = set()
    for t in clean_terms(terms or []):
        known.update(words(t))
    safe = [(h, w) for h, w in ((" ".join(str(h).split()), " ".join(str(w).split())) for h, w in pairs or [])
            if h and w and order_pair(h, w)]
    for _, written in safe:
        known.update(words(written))
    allowed = set(src) | FREE | known
    letters = [w for w in _alpha(out) if not any(c.isdigit() for c in w)]   # figures have their own guard
    af_src = [w for w in _without_corrections(src) if w in AFRIKAANS]
    distinct = set(af_src)
    brought_in = any(w not in set(src) | known for w in letters)
    if len(distinct) >= AF_MIN_DISTINCT and brought_in:    # gone with nothing new is a drop ("scratch that")
        missing = len(distinct - set(out)) / len(distinct)
        share_in = len(af_src) / max(1, len(_without_corrections(src)))
        share_out = sum(1 for w in out if w in AFRIKAANS) / max(1, len(out))
        if missing > AF_MISSING_MAX or share_out < AF_SHARE_MIN * share_in:
            return "translation"
    if len({w for w in out if w in AFRIKAANS} - allowed) >= AF_NEW_MAX:
        return "translation"                         # English turned into Afrikaans
    af, en, got = Counter(_without_corrections(src)), Counter(src), Counter(out)
    once = Counter(w for w, _ in _unstutter([(w, False) for w in _without_corrections(src) if w not in FILLERS]))
    for a, twins in TWINS.items():                   # one word swapped for its twin, either way
        if (got[a] < once[a] and any(got[e] > en[e] for e in twins)) or \
                (got[a] > af[a] and any(got[e] < en[e] for e in twins)):
            return "translation"
    if any(w not in allowed for w in letters):       # a spoken Afrikaans word gone, an unsaid word in
        keep = _kept_afrikaans(source)
        if any(got[a] < keep[a] for a in keep):
            return "translation"
    if not negation_ok(source, output):
        return "negation"
    added = [w for w in letters if w not in allowed]
    said = Counter(w for w in _alpha(src) if w not in FREE)
    extra = sum(n - said[w] for w, n in Counter(w for w in letters if w not in FREE).items()
                if said[w] and n > said[w])          # the speaker's own words used again: an echoed answer
    if len(added) + extra > max(ADDED_MAX, round(ADDED_FRAC * n_out)):
        return "added"
    if {_SAME.get(w, w) for w in out if w in ANSWERS} - {_SAME.get(w, w) for w in src}:
        return "added"                               # "Yes, ..." the speaker did not say: an answer
    spelt = respell(source, terms or [])
    for heard, written in safe:
        spelt = re.sub(_phrase(heard), lambda m, w=written: w, spelt, flags=re.I)
    if output.count("?") + _list_questions(spelt, output, known) < source.count("?"):
        return "added"                               # the speaker's question came back as a statement
    budget = said | Counter(w for w in _alpha(words(expand(spelt))) if w not in FREE) \
        | Counter(w for w in known if any(sounds_like(s, w) for s in set(src)))
    sentences = [[w for w in _alpha(words(s)) if w not in FREE and not any(c.isdigit() for c in w)]
                 for s in _SENTENCE.split(expand(output))]
    if _echoes(sentences, budget) or _echoes(sentences[::-1], budget):
        return "added"                               # a whole new sentence: an answer ("Paris.", "Done.")
    kept = {w for w in letters if w not in FREE}
    if any(w not in FREE and w not in FILLERS for w in _alpha(src)) and not kept & (set(src) | known) \
            and not any(c.isdigit() for c in output):         # a figure left is one the speaker said (figures_ok)
        return "dropped"
    return None


# ---------------------------------------------------------------- the cleaner

class Cleaner:
    """One dictation in, the cleaned text out. Thread-safe; clean() never raises and never blocks past
    its timeout. `last` describes the most recent clean() with no text in it."""

    def __init__(self, settings: dict, terms_fn: Callable[[], list[str]] | None = None,
                 pairs_fn: Callable[[], list[tuple[str, str]]] | None = None):
        self.settings = settings
        self.terms_fn = terms_fn
        self.pairs_fn = pairs_fn
        self.last: dict = {"ms": 0, "used": False, "reason": ""}
        self._lock = threading.Lock()
        self._warming = False
        self._send_think = True

    # -- settings

    def _cfg(self) -> dict:
        return dict((self.settings.get("dictation") or {}).get("cleanup") or {})

    def _url(self) -> str:
        return (self.settings.get("ollama") or {}).get("url", "http://127.0.0.1:11434").rstrip("/") + "/api/chat"

    def _terms(self) -> list[str]:
        if self.terms_fn is not None:
            return list(self.terms_fn() or [])
        return list(self.settings.get("vocabulary") or [])

    def _speaker(self) -> tuple[str, list[str]]:
        return str(self.settings.get("owner_name") or ""), list(_languages(self.settings.get("languages")))

    def _pairs(self) -> list[tuple[str, str]]:
        return list(self.pairs_fn() or []) if self.pairs_fn is not None else []

    def _body(self, cfg: dict, messages: list[dict], num_predict: int) -> dict:
        body = {
            "model": cfg.get("model"),
            "messages": messages,
            "stream": False,
            "keep_alive": cfg.get("keep_alive", "30m"),
            "options": {"temperature": 0, "top_k": 1, "repeat_penalty": 1.0, "num_ctx": NUM_CTX,
                        "num_predict": num_predict},
        }
        if self._send_think:
            body["think"] = False
        return body

    def _post(self, body: dict, timeout) -> dict:
        r = requests.post(self._url(), json=body, timeout=timeout, proxies=dict(NO_PROXY))
        if r.status_code == 400 and "think" in body and "think" in r.text.lower():
            self._send_think = False            # a model without a thinking switch refuses the field
            body = {k: v for k, v in body.items() if k != "think"}
            r = requests.post(self._url(), json=body, timeout=timeout, proxies=dict(NO_PROXY))
        if r.status_code == 404:
            raise LookupError(f"ollama: model {body.get('model')!r} not found")
        r.raise_for_status()
        return r.json()

    # -- warm

    def warm(self, wait: bool = False) -> None:
        """Load the model into VRAM (kept for keep_alive) and prime the prompt cache with the fixed rules
        and examples. Runs on its own thread; returns at once unless `wait`. Never raises."""
        try:
            cfg = self._cfg()
            if not cfg.get("enabled", True) or not cfg.get("model"):
                return
            with self._lock:
                if self._warming:
                    return
                self._warming = True
            t = threading.Thread(target=self._warm, args=(cfg,), name="scribe-cleanup-warm", daemon=True)
            t.start()
            if wait:
                t.join()
        except Exception as exc:
            log.warning("cleanup warm not started: %s", type(exc).__name__)
            with self._lock:
                self._warming = False

    def _warm(self, cfg: dict) -> None:
        t0 = time.perf_counter()
        try:
            msgs = build_messages("one two three four", self._terms(), None, *self._speaker())
            self._post(self._body(cfg, msgs, 1), WARM_TIMEOUT)
            log.info("cleanup model %s warm in %.0f ms", cfg.get("model"), (time.perf_counter() - t0) * 1000)
        except Exception as exc:
            log.warning("cleanup warm failed: %s", type(exc).__name__)
        finally:
            with self._lock:
                self._warming = False

    # -- clean

    def timeout_for(self, n_words: int) -> float:
        """timeout_s for a dictation up to 40 words, plus timeout_per_word_s for each word beyond."""
        cfg = self._cfg()
        base = float(cfg.get("timeout_s", 2.5))
        return base + float(cfg.get("timeout_per_word_s", 0.0)) * max(0, n_words - 40)

    def clean(self, text: str) -> str:
        """The cleaned dictation, or `text` unchanged when clean-up is off, the text is short, the model
        is slow or unreachable, or a guard rejects its reply. Never raises."""
        t0 = time.perf_counter()
        try:
            out, reason = self._clean(text)
        except Exception as exc:
            log.warning("cleanup failed: %s", type(exc).__name__)
            out, reason = text, "error"
        ms = int(round((time.perf_counter() - t0) * 1000))
        self.last = {"ms": ms, "used": reason == "ok", "reason": reason}
        if reason not in ("disabled", "short"):
            log.info("cleanup %s in %d ms", reason, ms)
        return out

    def _clean(self, text: str) -> tuple[str, str]:
        cfg = self._cfg()
        if not cfg.get("enabled", True) or not cfg.get("model"):
            return text, "disabled"
        if not isinstance(text, str) or len(text.split()) < MIN_WORDS:
            return text, "short"
        terms = clean_terms(self._terms())
        pairs = relevant_pairs(text, self._pairs())
        n = len(text.split())
        body = self._body(cfg, build_messages(collapse_repeats(text), terms, pairs, *self._speaker()),
                          min(PREDICT_MAX, PREDICT_BASE + PREDICT_PER_WORD * n))
        timeout = self.timeout_for(n)
        try:
            data = self._post(body, (min(CONNECT_S, timeout), timeout))
        except requests.ConnectTimeout:
            log.warning("cleanup: ollama not reachable")      # Windows takes ~2 s to refuse a closed port
            return text, "error"
        except requests.Timeout:
            self.warm()                  # most often a model that was unloaded: have it ready next time
            return text, "timeout"
        except (requests.RequestException, LookupError, ValueError) as exc:
            log.warning("cleanup request failed: %s", type(exc).__name__)
            return text, "error"
        if data.get("done_reason") == "length":
            return text, "guard:truncated"         # hit num_predict: the reply is cut off
        vocabulary = clean_terms(self.settings.get("vocabulary") or [])
        out = respell(tidy((data.get("message") or {}).get("content") or "", text), vocabulary)
        guard = check(text, out, terms, pairs)
        if guard:
            return text, f"guard:{guard}"
        return out, "ok"
