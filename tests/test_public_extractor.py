"""Public packaging must remain offline at import and reject altered extractor bytes."""
import hashlib
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / 'src/memory/state_controller.py'


class PublicExtractorTests(unittest.TestCase):
    def module(self):
        spec = importlib.util.spec_from_file_location('public_controller_under_test', PATH)
        module = importlib.util.module_from_spec(spec)
        with patch('urllib.request.urlopen') as fetch:
            spec.loader.exec_module(module)
            fetch.assert_not_called()
        return module

    def test_import_is_offline_and_pinned_bytes_are_accepted(self):
        module = self.module()
        raw = b'Synthetic extractor test fixture; not third-party source.'
        module.HISPA_SOURCE_SHA256 = hashlib.sha256(raw).hexdigest()
        with patch('urllib.request.urlopen') as fetch:
            fetch.return_value.__enter__.return_value.read.return_value = raw
            self.assertEqual(module._load_hispa_source(), raw.decode())
            self.assertIn('898d7f7b78ba8f934f70340d12e9c1183bb623e5', fetch.call_args.args[0])

    def test_modified_download_is_rejected_before_use(self):
        module = self.module()
        with patch('urllib.request.urlopen') as fetch:
            fetch.return_value.__enter__.return_value.read.return_value = b'wrong bytes'
            with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'):
                module._load_hispa_source()


if __name__ == '__main__':
    unittest.main()
