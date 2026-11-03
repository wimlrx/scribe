"""paste: clipboard round trip on the real clipboard (CF_UNICODETEXT only), and paste_text's flow with Ctrl+V patched out.

No test sends a key. Real-clipboard tests run only when the clipboard is empty or holds plain text, so nothing
richer (an image, an Excel range) can be lost; the user's text is put back afterwards, and only if the
clipboard still holds a test string. Test writes are kept out of Win+V history.
"""
from __future__ import annotations

import contextlib
import ctypes
import os
import unittest
from unittest import mock

from scribe import paste, winkeys

MARK = "scribe-test "
PLAIN_FORMATS = {paste.CF_TEXT, paste.CF_OEMTEXT, paste.CF_UNICODETEXT, paste.CF_LOCALE, paste.CF_CLOUD,
                 paste.CF_HISTORY, paste._RegisterClipboardFormatW("ExcludeClipboardContentFromMonitorProcessing")}

_real_write = paste._write_text_open


def _write_no_history(text: str, history: bool) -> bool:
    return _real_write(text, False)


class RealClipboard(unittest.TestCase):
    saved: str | None = None

    @classmethod
    def setUpClass(cls):
        try:
            formats = paste.clipboard_formats()
        except paste.ClipboardBusy:
            raise unittest.SkipTest("clipboard busy")
        rich = set(formats) - PLAIN_FORMATS
        if rich:
            raise unittest.SkipTest(f"clipboard holds {len(rich)} non-text formats; left untouched")
        cls.saved = paste.get_clipboard_text()

    @classmethod
    def tearDownClass(cls):
        try:
            current = paste.get_clipboard_text()
            if current is None or not current.startswith(MARK):
                return                                    # the user copied something meanwhile, or nothing written
            if cls.saved is not None:
                paste.set_clipboard_text(cls.saved, history=False)
            else:
                with paste._clipboard():
                    paste._EmptyClipboard()
        except paste.ClipboardBusy:
            pass

    def setUp(self):
        current = paste.get_clipboard_text()
        if current != self.saved and not (current or "").startswith(MARK):
            self.skipTest("clipboard changed by the user during the run")
        mock.patch.object(paste, "_write_text_open", side_effect=_write_no_history).start()
        mock.patch.object(winkeys, "wait_modifiers_released", return_value=True).start()
        mock.patch.object(paste, "_foreground_target", return_value=paste.Target(None, None, True)).start()
        self.addCleanup(mock.patch.stopall)

    def test_round_trip_unicode(self):
        for text in (MARK + "plain", MARK + "Alex sê dis 'n goeie idee — ê ë ô ü, R1 500 · 🎙", MARK + "a\r\nb\tc"):
            with self.subTest(text=text):
                self.assertTrue(paste.set_clipboard_text(text, history=False))
                self.assertEqual(paste.get_clipboard_text(), text)

    def test_flags_written(self):
        paste.set_clipboard_text(MARK + "flags", history=False)
        formats = paste.clipboard_formats()
        self.assertIn(paste.CF_UNICODETEXT, formats)
        self.assertIn(paste.CF_CLOUD, formats)
        self.assertIn(paste.CF_HISTORY, formats)

    def test_nul_stripped(self):
        paste.set_clipboard_text(MARK + "a\0b", history=False)
        self.assertEqual(paste.get_clipboard_text(), MARK + "ab")

    def test_paste_restores_previous_text(self):
        previous, dictated = MARK + "previous", MARK + "dictated"
        paste.set_clipboard_text(previous, history=False)
        seen = []

        def fake_send():
            seen.append(paste.get_clipboard_text())
            return True

        with mock.patch.object(paste, "_send_paste", side_effect=fake_send):
            self.assertTrue(paste.paste_text(dictated))
        self.assertEqual(seen, [dictated])
        self.assertEqual(paste.get_clipboard_text(), previous)

    def test_no_restore_when_clipboard_changed(self):
        paste.set_clipboard_text(MARK + "previous", history=False)

        def user_copies():
            paste.set_clipboard_text(MARK + "copied meanwhile", history=False)
            return True

        with mock.patch.object(paste, "_send_paste", side_effect=user_copies):
            self.assertTrue(paste.paste_text(MARK + "dictated"))
        self.assertEqual(paste.get_clipboard_text(), MARK + "copied meanwhile")

    def test_empty_clipboard_keeps_dictation(self):
        with paste._clipboard():
            paste._EmptyClipboard()
        with mock.patch.object(paste, "_send_paste", return_value=True):
            self.assertTrue(paste.paste_text(MARK + "dictated"))
        self.assertEqual(paste.get_clipboard_text(), MARK + "dictated")

    def test_failed_send_keeps_dictation(self):
        # SendInput refused (lock screen, UAC): the dictation must stay for a manual paste, never be restored over.
        paste.set_clipboard_text(MARK + "previous", history=False)
        with mock.patch.object(paste, "_send_paste", return_value=False), self.assertLogs("scribe.paste", "WARNING"):
            self.assertFalse(paste.paste_text(MARK + "dictated"))
        self.assertEqual(paste.get_clipboard_text(), MARK + "dictated")

    def test_target_that_never_reads_keeps_dictation(self):
        # Ctrl+V went out but nobody asked for the text: restoring would lose the dictation.
        paste.set_clipboard_text(MARK + "previous", history=False)
        with mock.patch.object(paste, "RENDER_WAIT_S", 0.2), \
                mock.patch.object(paste, "_send_paste", return_value=True), \
                self.assertLogs("scribe.paste", "WARNING"):
            self.assertFalse(paste.paste_text(MARK + "dictated"))
        self.assertEqual(paste.get_clipboard_text(), MARK + "dictated")

    def test_restore_waits_for_the_targets_read(self):
        # The old contents go back only after the text was rendered for the reader, never on a fixed timer.
        paste.set_clipboard_text(MARK + "previous", history=False)
        order = []

        def fake_send():
            order.append(("sent", paste.get_clipboard_text()))
            return True

        real_restore = paste._restore

        def spy_restore(previous, ours):
            order.append(("restore", previous.text()))
            real_restore(previous, ours)

        with mock.patch.object(paste, "_send_paste", side_effect=fake_send), \
                mock.patch.object(paste, "_restore", side_effect=spy_restore):
            self.assertTrue(paste.paste_text(MARK + "dictated"))
        self.assertEqual(order, [("sent", MARK + "dictated"), ("restore", MARK + "previous")])

    def test_every_format_restored(self):
        # A clipboard with a second (private, registered) format comes back whole, not flattened to text.
        fmt = paste._RegisterClipboardFormatW("Scribe Test Format")
        blob = bytes(range(256)) * 4
        with paste._clipboard():
            paste._write_text_open(MARK + "rich", history=False)
            self.assertTrue(paste._set_global(fmt, blob))

        def fake_send():
            paste.get_clipboard_text()
            return True

        with mock.patch.object(paste, "_send_paste", side_effect=fake_send):
            self.assertTrue(paste.paste_text(MARK + "dictated"))
        self.assertEqual(paste.get_clipboard_text(), MARK + "rich")
        with paste._clipboard():
            self.assertIn(fmt, paste._formats_open())
            handle = paste._GetClipboardData(fmt)
            ptr = paste._GlobalLock(handle)
            try:
                self.assertEqual(ctypes.string_at(ptr, len(blob)), blob)
            finally:
                paste._GlobalUnlock(handle)

    def test_modifiers_held_leaves_text_and_sends_nothing(self):
        paste.set_clipboard_text(MARK + "previous", history=False)
        send = mock.patch.object(paste, "_send_paste").start()
        with mock.patch.object(winkeys, "wait_modifiers_released", return_value=False), \
                self.assertLogs("scribe.paste", "WARNING"):
            self.assertFalse(paste.paste_text(MARK + "dictated"))
        send.assert_not_called()
        self.assertEqual(paste.get_clipboard_text(), MARK + "dictated")


class FakeOwner:
    """Stands in for the message-only window: `reads` says what happens after Ctrl+V."""

    reads = "target"           # "target" | "other" | "none"

    def __init__(self, text, target_pid):
        self.text, self.target_pid = text, target_pid
        self.hwnd = 1
        self.armed = False
        self.renders = 0
        self.target_read = False
        self.lost = False
        self.closed = False

    def pump(self, until, timeout):
        if self.armed and self.renders == 0 and FakeOwner.reads != "none":
            self.renders = 1
            self.target_read = FakeOwner.reads == "target"
        return until()

    def close(self):
        self.closed = True


class Flow(unittest.TestCase):
    """paste_text decisions with the clipboard itself patched out."""

    def patch(self, name, **kw):
        return mock.patch.object(paste, name, **kw).start()

    def setUp(self):
        self.addCleanup(mock.patch.stopall)
        FakeOwner.reads = "target"
        self.waits = mock.patch.object(winkeys, "wait_modifiers_released", return_value=True).start()
        self.patch("_clipboard", side_effect=lambda *a: contextlib.nullcontext())
        self.write = self.patch("_write_text_open", return_value=True)
        self.patch("_EmptyClipboard", return_value=1)
        self.set_data = self.patch("_SetClipboardData", return_value=1)
        self.patch("_flags_open")
        self.send = self.patch("_send_paste", return_value=True)
        self.restore = self.patch("_restore")
        self.leave = self.patch("_leave")
        self.settle = self.patch("_settle", return_value=True)
        self.patch("_Owner", side_effect=FakeOwner)
        self.target = self.patch("_foreground_target", return_value=paste.Target(5, 77, True))
        mock.patch.object(paste.time, "sleep").start()

    def snapshot(self, text="old"):
        snap = paste.Snapshot()
        if text is not None:
            snap.items.append((paste.CF_UNICODETEXT, "global", (text + "\0").encode("utf-16-le")))
        else:
            snap.items.append((paste.CF_DIB, "global", b"BM..."))        # an image only
        return self.patch("_snapshot_open", return_value=snap)

    def test_image_clipboard_is_restored_too(self):
        self.snapshot(None)
        self.assertTrue(paste.paste_text("hallo"))
        self.set_data.assert_called_once_with(paste.CF_UNICODETEXT, None)    # offered by delayed rendering
        self.send.assert_called_once()
        self.restore.assert_called_once()
        self.assertEqual(self.restore.call_args.args[0].items[0][0], paste.CF_DIB)

    def test_text_clipboard_restored_after_the_targets_read(self):
        self.snapshot("old")
        self.assertTrue(paste.paste_text("new"))
        self.assertEqual(self.restore.call_args.args[0].text(), "old")
        self.assertEqual(self.restore.call_args.args[1], "new")
        self.settle.assert_not_called()

    def test_read_by_someone_else_waits_for_the_target_queue(self):
        self.snapshot("old")
        FakeOwner.reads = "other"
        self.assertTrue(paste.paste_text("new"))
        self.settle.assert_called_once_with(5)
        self.restore.assert_called_once()

    def test_hung_target_keeps_dictation(self):
        self.snapshot("old")
        FakeOwner.reads = "other"
        self.settle.return_value = False
        with self.assertLogs("scribe.paste", "WARNING"):
            self.assertFalse(paste.paste_text("new"))
        self.restore.assert_not_called()
        self.leave.assert_called_once_with("new")

    def test_unread_text_is_never_restored_over(self):
        self.snapshot("old")
        FakeOwner.reads = "none"
        with self.assertLogs("scribe.paste", "WARNING"):
            self.assertFalse(paste.paste_text("new"))
        self.restore.assert_not_called()
        self.leave.assert_called_once_with("new")

    def test_failed_send_keeps_dictation(self):
        self.snapshot("old")
        self.send.return_value = False
        with self.assertLogs("scribe.paste", "WARNING"):
            self.assertFalse(paste.paste_text("new"))
        self.restore.assert_not_called()
        self.leave.assert_called_once_with("new")

    def test_empty_clipboard_writes_text_directly(self):
        self.patch("_snapshot_open", return_value=None)
        self.assertTrue(paste.paste_text("hallo"))
        self.write.assert_called_once_with("hallo", history=True)
        self.restore.assert_not_called()

    def test_elevated_target_gets_no_keys(self):
        self.snapshot("old")
        self.target.return_value = paste.Target(5, 77, False, "foreground runs elevated")
        with self.assertLogs("scribe.paste", "WARNING"):
            self.assertFalse(paste.paste_text("new"))
        self.send.assert_not_called()
        self.restore.assert_not_called()
        self.leave.assert_called_once_with("new", only_if_ours=False)

    def test_own_hold_is_waited_out_without_the_two_second_limit(self):
        self.patch("_snapshot_open", return_value=None)
        holding = iter([True] * 300 + [False])          # 300 polls of 20 ms: past the old 2 s limit
        seen = []

        def hold():
            seen.append(1)
            return next(holding)

        self.assertTrue(paste.paste_text("clip one", hold_active=hold))
        self.assertEqual(len(seen), 301)
        self.waits.assert_called_once()
        self.send.assert_called_once()

    def test_empty_text_touches_nothing(self):
        for text in ("", "\0", None):
            self.assertFalse(paste.paste_text(text))
        self.write.assert_not_called()
        self.send.assert_not_called()

    def test_write_failure_sends_nothing(self):
        self.patch("_snapshot_open", return_value=None)
        self.write.return_value = False
        with self.assertLogs("scribe.paste", "ERROR"):
            self.assertFalse(paste.paste_text("new"))
        self.send.assert_not_called()

    def test_ctrl_v_sequence(self):
        mock.patch.stopall()
        with mock.patch.object(winkeys, "send_keys", return_value=True) as sk:
            self.assertTrue(paste._send_paste())
        self.assertEqual(list(sk.call_args.args[0]), [(winkeys.VK_CONTROL, True), (0x56, True), (0x56, False),
                                                      (winkeys.VK_CONTROL, False)])


class TargetWindow(unittest.TestCase):
    def test_own_integrity_is_readable(self):
        level = paste.process_integrity()
        self.assertIsInstance(level, int)
        self.assertGreaterEqual(level, 0x1000)
        self.assertEqual(paste.process_integrity(os.getpid()), level)

    def test_no_foreground_window_refuses(self):
        with mock.patch.object(paste, "_GetForegroundWindow", return_value=None):
            self.assertFalse(paste._foreground_target().ok)

    def test_higher_or_unreadable_integrity_refuses(self):
        with mock.patch.object(paste, "_GetForegroundWindow", return_value=123), \
                mock.patch.object(paste, "_window_pid", return_value=4242):
            for theirs, ok in ((0x2000, True), (0x1000, True), (0x3000, False), (None, False)):
                with self.subTest(theirs=theirs), mock.patch.object(
                        paste, "process_integrity", side_effect=lambda pid=None, t=theirs: 0x2000 if pid is None else t):
                    self.assertEqual(paste._foreground_target().ok, ok)


class OpenRetry(unittest.TestCase):
    def setUp(self):
        self.addCleanup(mock.patch.stopall)
        self.sleep = mock.patch.object(paste.time, "sleep").start()
        self.close = mock.patch.object(paste, "_CloseClipboard", return_value=1).start()
        mock.patch.object(paste, "_read_text_open", return_value="x").start()

    def test_succeeds_on_tenth_try(self):
        opener = mock.patch.object(paste, "_OpenClipboard", side_effect=[0] * 9 + [1]).start()
        self.assertEqual(paste.get_clipboard_text(), "x")
        self.assertEqual(opener.call_count, 10)
        self.assertEqual([c.args for c in self.sleep.call_args_list], [(paste.OPEN_SLEEP_S,)] * 9)
        self.close.assert_called_once()

    def test_gives_up_after_ten(self):
        opener = mock.patch.object(paste, "_OpenClipboard", return_value=0).start()
        with self.assertRaises(paste.ClipboardBusy):
            paste.get_clipboard_text()
        self.assertEqual(opener.call_count, 10)
        self.close.assert_not_called()

    def test_busy_clipboard_paste_fails_without_keys(self):
        mock.patch.object(paste, "_OpenClipboard", return_value=0).start()
        mock.patch.object(winkeys, "wait_modifiers_released", return_value=True).start()
        mock.patch.object(paste, "_foreground_target", return_value=paste.Target(None, None, True)).start()
        send = mock.patch.object(paste, "_send_paste").start()
        with self.assertLogs("scribe.paste", "ERROR"):
            self.assertFalse(paste.paste_text("hallo"))
        send.assert_not_called()

    def test_closed_even_on_error(self):
        mock.patch.object(paste, "_OpenClipboard", return_value=1).start()
        mock.patch.object(paste, "_read_text_open", side_effect=RuntimeError).start()
        with self.assertRaises(RuntimeError):
            paste.get_clipboard_text()
        self.close.assert_called_once()


class Writes(unittest.TestCase):
    def test_history_and_cloud_flags(self):
        with mock.patch.object(paste, "_EmptyClipboard", return_value=1), \
                mock.patch.object(paste, "_set_global", return_value=True) as sg:
            paste._write_text_open("hi", history=True)
            self.assertEqual([c.args[0] for c in sg.call_args_list], [paste.CF_UNICODETEXT, paste.CF_CLOUD])
            sg.reset_mock()
            paste._write_text_open("hi", history=False)
            self.assertEqual([c.args[0] for c in sg.call_args_list],
                             [paste.CF_UNICODETEXT, paste.CF_CLOUD, paste.CF_HISTORY])
            self.assertEqual(sg.call_args_list[0].args[1], "hi\0".encode("utf-16-le"))
            self.assertEqual(sg.call_args_list[1].args[1], b"\0\0\0\0")

    def test_every_win32_function_declared(self):
        for f in paste.WIN32_FUNCTIONS:
            with self.subTest(f=f.__name__):
                self.assertIsNotNone(f.argtypes)
                self.assertIsNotNone(f.restype)
        self.assertIs(paste._GlobalLock.restype, ctypes.c_void_p)
        self.assertIs(paste._GetClipboardData.restype, ctypes.wintypes.HANDLE)
        self.assertTrue(paste.CF_CLOUD and paste.CF_HISTORY)


if __name__ == "__main__":
    unittest.main()
