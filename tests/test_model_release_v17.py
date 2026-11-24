"""Deterministic offline unit tests; no real GitHub release is created."""
import hashlib
import json
from pathlib import Path

import pytest

from omega_geodepth.model_release import (
    ASSET_NAMES, GitHubReleaseClient, _checked_asset,
    main, publish_release, validate_local_artifacts,
)


def fixture_inventory(tmp_path):
    pending = []
    for i, name in enumerate(ASSET_NAMES):
        contents = bytes([i + 1]) * (i + 3)
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        pending.append({'name': name, 'path': name, 'bytes': len(contents),
                        'sha256': hashlib.sha256(contents).hexdigest()})
    inv = tmp_path / 'models' / 'repository_inventory.json'
    inv.write_text(json.dumps({'schema': 'geodepth.repository-model-inventory/1',
                               'pending': pending}), encoding='utf-8')
    return inv


def test_local_all_five_files_checked_by_sha(tmp_path):
    inv = fixture_inventory(tmp_path)
    result = validate_local_artifacts(tmp_path, inv)
    assert len(result) == 5
    assert {x.asset_name for x in result} == set(ASSET_NAMES.values())


def test_dry_run_never_contacts_github(tmp_path, capsys):
    fixture_inventory(tmp_path)
    assert main(['--root', str(tmp_path)]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc['local_preflight'] == 'PASS'
    assert doc['github'] == 'NOT_ATTEMPTED_DRY_RUN'


def test_missing_model_aborts_preflight(tmp_path):
    inv = fixture_inventory(tmp_path)
    (tmp_path / next(iter(ASSET_NAMES))).unlink()
    with pytest.raises(FileNotFoundError, match='MODEL_FILE_MISSING'):
        validate_local_artifacts(tmp_path, inv)


def test_modified_model_aborts_preflight(tmp_path):
    inv = fixture_inventory(tmp_path)
    model = tmp_path / next(iter(ASSET_NAMES))
    model.write_bytes(b'q' * model.stat().st_size)
    with pytest.raises(ValueError, match='MODEL_SHA256_MISMATCH'):
        validate_local_artifacts(tmp_path, inv)


def test_symlink_does_not_bypass_inventory(tmp_path):
    inv = fixture_inventory(tmp_path)
    model = tmp_path / next(iter(ASSET_NAMES))
    original = model.read_bytes()
    model.unlink()
    actual = tmp_path / 'rogue.bin'
    actual.write_bytes(original)
    model.symlink_to(actual)
    with pytest.raises(ValueError, match='UNSAFE_MODEL_SYMLINK'):
        validate_local_artifacts(tmp_path, inv)


@pytest.mark.parametrize('bad', ['../x.safetensors', 'models/da3/nope.bin', 'models/da3/model.safetensors'])
def test_reject_extra_or_duplicate_paths(tmp_path, bad):
    inv = fixture_inventory(tmp_path)
    doc = json.loads(inv.read_text())
    doc['pending'][-1]['path'] = bad
    inv.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='UNEXPECTED_MODEL_PATH'):
        validate_local_artifacts(tmp_path, inv)


def test_reject_untrusted_token_and_asset_digest(tmp_path):
    with pytest.raises(ValueError, match='GITHUB_TOKEN_REQUIRED'):
        GitHubReleaseClient('abc\nInjected: yes')
    inv = fixture_inventory(tmp_path)
    artifact = validate_local_artifacts(tmp_path, inv)[0]
    with pytest.raises(ValueError, match='INTEGRITY_UNVERIFIED'):
        _checked_asset({'name': artifact.asset_name, 'size': artifact.size,
                        'state': 'uploaded', 'digest': None}, artifact)


class FakeReleaseClient:
    """Offline API contract stub; proves release transitions, NOT live GitHub integration."""
    def __init__(self, preexisting=False, moved=False, bad_digest=False):
        self.calls = []
        self.uploads = []
        self.moved = moved
        self.bad_digest = bad_digest
        self.main_calls = 0
        self.release = {'id': 17, 'tag_name': 'geodepth-da3-models-v0.16.0', 'draft': True}
        self.preexisting = preexisting

    def api(self, method, path, data=None, *, optional_404=False):
        self.calls.append((method, path, data))
        if path.endswith('/git/ref/heads/main'):
            self.main_calls += 1
            return {'object': {'sha': 'f' * 40 if self.moved and self.main_calls == 2 else 'e' * 40}}
        if '/releases/tags/' in path:
            return self.release if self.preexisting else None
        if method == 'POST' and path.endswith('/releases'):
            assert data['draft'] is True and data['target_commitish'] == 'e' * 40
            return self.release
        if path.endswith('/assets?per_page=100'):
            return []
        if method == 'PATCH':
            assert data == {'draft': False}
            return {'draft': False, 'tag_name': self.release['tag_name'],
                    'html_url': 'https://github.com/mojealterego/GeoDepth/releases/tag/geodepth-da3-models-v0.16.0'}
        raise AssertionError((method, path, data))

    def upload(self, release_id, artifact):
        assert release_id == 17
        self.uploads.append(artifact.asset_name)
        return {'name': artifact.asset_name, 'size': artifact.size, 'state': 'uploaded',
                'digest': 'sha256:' + ('0' * 64 if self.bad_digest else artifact.sha256)}


def test_release_draft_upload_and_publish(tmp_path):
    inv = fixture_inventory(tmp_path)
    client = FakeReleaseClient()
    result = publish_release(client, validate_local_artifacts(tmp_path, inv),
                             'geodepth-da3-models-v0.16.0')
    assert result['verified_sha256'] and len(client.uploads) == 5
    assert client.calls[-1][0] == 'PATCH'


def test_reject_moved_main_before_publish(tmp_path):
    inv = fixture_inventory(tmp_path)
    client = FakeReleaseClient(moved=True)
    with pytest.raises(RuntimeError, match='MAIN_MOVED'):
        publish_release(client, validate_local_artifacts(tmp_path, inv),
                        'geodepth-da3-models-v0.16.0')
    assert not any(method == 'PATCH' for method, *_ in client.calls)


def test_reject_bad_remote_digest_before_publish(tmp_path):
    inv = fixture_inventory(tmp_path)
    client = FakeReleaseClient(bad_digest=True)
    with pytest.raises(ValueError, match='INTEGRITY_UNVERIFIED'):
        publish_release(client, validate_local_artifacts(tmp_path, inv),
                        'geodepth-da3-models-v0.16.0')
    assert not any(method == 'PATCH' for method, *_ in client.calls)


def test_empty_token_fails_closed_without_network():
    with pytest.raises(ValueError, match='GITHUB_TOKEN_REQUIRED'):
        GitHubReleaseClient('')


def test_symlink_root_rejected_by_cli(tmp_path):
    original = tmp_path / 'real'
    original.mkdir()
    fixture_inventory(original)
    link = tmp_path / 'link'
    link.symlink_to(original, target_is_directory=True)
    with pytest.raises(ValueError, match='UNSAFE_MODEL_ROOT'):
        main(['--root', str(link)])


def test_symlink_manifest_parent_rejected(tmp_path):
    inv = fixture_inventory(tmp_path)
    path = tmp_path / 'models'
    real = tmp_path / 'actual_models'
    path.rename(real)
    path.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match='UNSAFE_MODEL_INVENTORY_PATH'):
        validate_local_artifacts(tmp_path, path / 'repository_inventory.json')


def test_resume_existing_draft(tmp_path):
    inv = fixture_inventory(tmp_path)
    client = FakeReleaseClient(preexisting=True)
    publish_release(client, validate_local_artifacts(tmp_path, inv),
                    'geodepth-da3-models-v0.16.0')
    assert len(client.uploads) == 5
    assert not any(method == 'POST' and path.endswith('/releases') for method, path, _ in client.calls)
