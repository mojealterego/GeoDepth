"""RED/GREEN: integrate the user-supplied upstream DA3 source securely and reproducibly."""
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
import zipfile

from omega_geodepth.da3_source import (
    BUNDLED_DA3_SHA256,
    inspect_da3_source_archive,
    extract_da3_source,
    verify_extracted_da3_source,
)

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'vendor' / 'Depth-Anything-3-main.zip'


def fake_zip(files: dict[str,bytes]) -> bytes:
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w',compression=zipfile.ZIP_DEFLATED) as f:
        for p,contents in files.items(): f.writestr(p,contents)
    return b.getvalue()


class DA3SourceTests(unittest.TestCase):
    def test_01_user_upstream_source_has_expected_integrity(self):
        report=inspect_da3_source_archive(ARCHIVE)
        self.assertEqual(report['sha256'],BUNDLED_DA3_SHA256)
        self.assertTrue(report['source_only'])
        self.assertGreater(report['entry_count'],100)
        self.assertTrue(report['api_contract_verified'])

    def test_02_bundle_is_reproducibly_extracted_and_tamper_detected(self):
        with tempfile.TemporaryDirectory() as td:
            directory=Path(td)/'official'
            result=extract_da3_source(ARCHIVE,directory)
            self.assertEqual(result['sha256'],BUNDLED_DA3_SHA256)
            self.assertTrue((directory/'src/depth_anything_3/api.py').is_file())
            self.assertTrue(verify_extracted_da3_source(directory)['ok'])
            with (directory/'src/depth_anything_3/api.py').open('a') as f:f.write('\n# altered\n')
            with self.assertRaisesRegex(ValueError,'DA3_SOURCE_TAMPERED'):
                verify_extracted_da3_source(directory)

    def test_03_prevent_overwrite_existing_destination(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td)/'target';d.mkdir()
            (d/'user-file').write_text('DO_NOT_CHANGE')
            with self.assertRaisesRegex(FileExistsError,'DA3_SOURCE_TARGET_EXISTS'):
                extract_da3_source(ARCHIVE,d)
            self.assertEqual((d/'user-file').read_text(),'DO_NOT_CHANGE')

    def test_04_reject_traversal_even_with_custom_hash(self):
        payload=fake_zip({'Depth-Anything-3-main/../../escape.txt':b'bad'})
        with tempfile.TemporaryDirectory() as td:
            z=Path(td)/'test.zip';z.write_bytes(payload)
            with self.assertRaisesRegex(ValueError,'DA3_SOURCE_UNSAFE_PATH'):
                inspect_da3_source_archive(z,expected_sha256=hashlib.sha256(payload).hexdigest())
            self.assertFalse((Path(td)/'escape.txt').exists())

    def test_05_reject_duplicate_entries(self):
        with tempfile.TemporaryDirectory() as td:
            z=Path(td)/'test.zip'
            with zipfile.ZipFile(z,'w') as f:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',UserWarning)
                    f.writestr('Depth-Anything-3-main/a',b'1')
                    f.writestr('Depth-Anything-3-main/a',b'2')
            with self.assertRaisesRegex(ValueError,'DA3_SOURCE_DUPLICATE'):
                inspect_da3_source_archive(z,expected_sha256=hashlib.sha256(z.read_bytes()).hexdigest())

    def test_06_reject_symlink_entries(self):
        with tempfile.TemporaryDirectory() as td:
            z=Path(td)/'test.zip'
            with zipfile.ZipFile(z,'w') as f:
                info=zipfile.ZipInfo('Depth-Anything-3-main/link')
                info.create_system=3
                info.external_attr=(0o120777<<16)
                f.writestr(info,b'../../untrusted')
            with self.assertRaisesRegex(ValueError,'DA3_SOURCE_SYMLINK'):
                inspect_da3_source_archive(z,expected_sha256=hashlib.sha256(z.read_bytes()).hexdigest())

    def test_07_reject_fake_hash(self):
        with self.assertRaisesRegex(ValueError,'DA3_SOURCE_ARCHIVE_HASH_MISMATCH'):
            inspect_da3_source_archive(ARCHIVE,expected_sha256='0'*64)

    def test_08_reject_missing_extraction_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex((ValueError,FileNotFoundError),'DA3_SOURCE_'):
                verify_extracted_da3_source(Path(td))

    def test_09_cannot_inject_undeclared_python_files(self):
        with tempfile.TemporaryDirectory() as td:
            directory=Path(td)/'official';extract_da3_source(ARCHIVE,directory)
            other=directory/'src/depth_anything_3/custom_backdoor.py';other.write_text('print(1)')
            with self.assertRaisesRegex(ValueError,'DA3_SOURCE_UNDECLARED_FILE'):
                verify_extracted_da3_source(directory)

    def test_10_bundle_provenance_explicit(self):
        report=inspect_da3_source_archive(ARCHIVE)
        self.assertIn('user_supplied',report['source_provenance'])
        self.assertFalse(report['upstream_commit_verified'])

if __name__=='__main__':unittest.main()
