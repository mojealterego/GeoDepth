"""Auditable, fail-closed publication of verified DA3 weights as GitHub Release assets.

No network calls occur during preparation. Publishing requires --publish and
GEODEPTH_GITHUB_TOKEN. Assets are immutable: mismatches block the release.
GitHub Release assets reside in the repository, not its main Git tree.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

REPOSITORY = 'mojealterego/GeoDepth'
SCHEMA = 'geodepth.repository-model-inventory/1'
ASSET_NAMES = {
    'models/da3/model.safetensors': 'da3-small-fp32.safetensors',
    'models/da3_base/model.safetensors': 'da3-base-fp32.safetensors',
    'models/da3_profiles/fp16/model.safetensors': 'da3-small-fp16.safetensors',
    'models/da3_profiles/aq4/model.safetensors': 'da3-small-aq4.safetensors',
    'models/da3_profiles/aq6/model.safetensors': 'da3-small-aq6.safetensors',
}
MAX_ASSET_BYTES = 2_000_000_000
CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class Artifact:
    path: Path
    repo_path: str
    asset_name: str
    size: int
    sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_local_artifacts(root: Path, inventory: Path) -> list[Artifact]:
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('UNSAFE_MODEL_ROOT')
    if inventory.is_symlink() or not inventory.is_file() or inventory.stat().st_size > 100_000:
        raise ValueError('INVALID_MODEL_INVENTORY')
    document = json.loads(inventory.read_text(encoding='utf-8'))
    if not isinstance(document, dict) or document.get('schema') != SCHEMA:
        raise ValueError('INVALID_MODEL_INVENTORY_SCHEMA')
    pending = document.get('pending')
    if not isinstance(pending, list) or len(pending) != len(ASSET_NAMES):
        raise ValueError('INCOMPLETE_DA3_INVENTORY')
    artifacts: list[Artifact] = []
    found = set()
    for item in pending:
        if not isinstance(item, dict):
            raise ValueError('INVALID_MODEL_ENTRY')
        name = item.get('path')
        if not isinstance(name, str) or name not in ASSET_NAMES or name in found:
            raise ValueError('UNEXPECTED_MODEL_PATH')
        if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:
            raise ValueError('UNSAFE_MODEL_PATH')
        found.add(name)
        size = item.get('bytes')
        sha = item.get('sha256')
        if type(size) is not int or not 0 < size <= MAX_ASSET_BYTES:
            raise ValueError('INVALID_MODEL_SIZE')
        if not isinstance(sha, str) or re.fullmatch(r'[0-9a-f]{64}', sha) is None:
            raise ValueError('INVALID_MODEL_HASH')
        relative = PurePosixPath(name)
        target = root.joinpath(*relative.parts)
        for parent in (root, *list(target.parents)[:-1], target):
            if parent.is_symlink():
                raise ValueError('UNSAFE_MODEL_SYMLINK')
        if not target.is_file() or not stat.S_ISREG(target.stat().st_mode):
            raise FileNotFoundError(f'MODEL_FILE_MISSING: {name}')
        if target.stat().st_size != size:
            raise ValueError(f'MODEL_SIZE_MISMATCH: {name}')
        if sha256_file(target) != sha:
            raise ValueError(f'MODEL_SHA256_MISMATCH: {name}')
        artifacts.append(Artifact(target, name, ASSET_NAMES[name], size, sha))
    if found != set(ASSET_NAMES):
        raise ValueError('INCOMPLETE_DA3_INVENTORY')
    return artifacts


class GitHubReleaseClient:
    def __init__(self, token: str, *, timeout: float = 60.0):
        if not token or not token.strip() or any(c in token for c in '\r\n'):
            raise ValueError('GITHUB_TOKEN_REQUIRED')
        self._token = token
        self._timeout = timeout

    def api(self, method: str, path: str, data: dict | None = None, *, optional_404: bool = False) -> dict | None:
        if not path.startswith(f'/repos/{REPOSITORY}/') or '://' in path:
            raise ValueError('UNSAFE_GITHUB_API_PATH')
        payload = None if data is None else json.dumps(data, separators=(',', ':')).encode('utf-8')
        headers = {'Authorization': f'Bearer {self._token}', 'Accept': 'application/vnd.github+json',
                   'X-GitHub-Api-Version': '2022-11-28', 'User-Agent': 'GeoDepth-model-release/1'}
        if payload is not None:
            headers['Content-Type'] = 'application/json'
        for attempt in range(4):
            req = Request('https://api.github.com' + path, data=payload, headers=headers, method=method)
            try:
                with urlopen(req, timeout=self._timeout) as response:
                    return json.loads(response.read(2_000_000))
            except HTTPError as err:
                if optional_404 and err.code == 404:
                    return None
                if err.code in (429, 500, 502, 503, 504) and attempt < 3:
                    time.sleep(min(8, 2 ** attempt))
                    continue
                raise RuntimeError(f'GITHUB_API_HTTP_{err.code}') from None
            except (URLError, TimeoutError) as err:
                if attempt == 3:
                    raise RuntimeError('GITHUB_API_NETWORK_FAILURE') from err
                time.sleep(min(8, 2 ** attempt))
        raise RuntimeError('GITHUB_API_RETRY_EXHAUSTED')

    def upload(self, release_id: int, artifact: Artifact) -> dict:
        if type(release_id) is not int or release_id <= 0:
            raise ValueError('INVALID_RELEASE_ID')
        path = (f'/repos/{REPOSITORY}/releases/{release_id}/assets?name='
                + quote(artifact.asset_name, safe=''))
        # Stream the original file. No Base64 expansion or 500-MB in-memory body.
        connection = http.client.HTTPSConnection('uploads.github.com', timeout=self._timeout)
        try:
            connection.putrequest('POST', path)
            for name, value in {
                'Authorization': f'Bearer {self._token}',
                'Accept': 'application/vnd.github+json',
                'Content-Type': 'application/octet-stream',
                'Content-Length': str(artifact.size),
                'User-Agent': 'GeoDepth-model-release/1',
                'X-GitHub-Api-Version': '2022-11-28',
            }.items():
                connection.putheader(name, value)
            connection.endheaders()
            digest = hashlib.sha256()
            sent = 0
            with artifact.path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(CHUNK_BYTES), b''):
                    connection.send(chunk)
                    digest.update(chunk)
                    sent += len(chunk)
            if sent != artifact.size or digest.hexdigest() != artifact.sha256:
                raise RuntimeError('LOCAL_FILE_CHANGED_DURING_UPLOAD')
            response = connection.getresponse()
            response_body = response.read(2_000_000)
            if response.status != 201:
                raise RuntimeError(f'GITHUB_UPLOAD_HTTP_{response.status}')
            return json.loads(response_body)
        finally:
            connection.close()


def _checked_asset(item: dict, artifact: Artifact) -> None:
    if item.get('name') != artifact.asset_name or item.get('state') != 'uploaded':
        raise ValueError(f'RELEASE_ASSET_STATE_MISMATCH: {artifact.asset_name}')
    if item.get('size') != artifact.size or item.get('digest') != f'sha256:{artifact.sha256}':
        raise ValueError(f'RELEASE_ASSET_INTEGRITY_UNVERIFIED: {artifact.asset_name}')


def publish_release(client: GitHubReleaseClient, artifacts: list[Artifact], tag: str) -> dict:
    if re.fullmatch(r'geodepth-da3-models-v\d+\.\d+\.\d+', tag) is None:
        raise ValueError('INVALID_RELEASE_TAG')
    if len(artifacts) != len(ASSET_NAMES) or {a.asset_name for a in artifacts} != set(ASSET_NAMES.values()):
        raise ValueError('INCOMPLETE_ASSET_SET')
    prefix = f'/repos/{REPOSITORY}'
    head = client.api('GET', prefix + '/git/ref/heads/main')['object']['sha']
    release = client.api('GET', prefix + '/releases/tags/' + quote(tag, safe=''), optional_404=True)
    if release is None:
        release = client.api('POST', prefix + '/releases', {'tag_name': tag, 'target_commitish': head,
                                'name': 'GeoDepth verified DA3 checkpoint distribution', 'draft': True,
                                'prerelease': True, 'generate_release_notes': False,
                                'body': 'SHA-256 verified checkpoint files. See models/repository_inventory.json.'})
    if release.get('tag_name') != tag or not release.get('draft'):
        raise ValueError('RELEASE_NOT_EDITABLE_DRAFT')
    release_id = release['id']
    all_assets = client.api('GET', prefix + f'/releases/{release_id}/assets?per_page=100')
    if not isinstance(all_assets, list):
        raise ValueError('INVALID_RELEASE_ASSET_LIST')
    remote = {a['name']: a for a in all_assets if a.get('name') in ASSET_NAMES.values()}
    for artifact in artifacts:
        current = remote.get(artifact.asset_name)
        if current is not None:
            _checked_asset(current, artifact)
        else:
            uploaded = client.upload(release_id, artifact)
            _checked_asset(uploaded, artifact)
            remote[artifact.asset_name] = uploaded
    for artifact in artifacts:
        _checked_asset(remote[artifact.asset_name], artifact)
    current_head = client.api('GET', prefix + '/git/ref/heads/main')['object']['sha']
    if current_head != head:
        raise RuntimeError('MAIN_MOVED_DURING_RELEASE')
    published = client.api('PATCH', prefix + f'/releases/{release_id}', {'draft': False})
    if published.get('draft') or published.get('tag_name') != tag:
        raise RuntimeError('RELEASE_PUBLICATION_NOT_CONFIRMED')
    return {'release_url': published.get('html_url'), 'release_id': release_id,
            'main_sha': head, 'assets': len(remote), 'verified_sha256': True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Verify DA3 artifacts, optionally publish GitHub Release')
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--inventory', type=Path, default=Path('models/repository_inventory.json'))
    parser.add_argument('--release-tag', default='geodepth-da3-models-v0.16.0')
    parser.add_argument('--publish', action='store_true', help='Explicitly create/upload/publish repository Release')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    inventory = args.inventory if args.inventory.is_absolute() else root / args.inventory
    artifacts = validate_local_artifacts(root, inventory)
    report: dict[str, Any] = {'local_preflight': 'PASS', 'count': len(artifacts),
        'total_bytes': sum(x.size for x in artifacts), 'assets': [x.asset_name for x in artifacts]}
    if args.publish:
        token = os.environ.get('GEODEPTH_GITHUB_TOKEN', '')
        report['github'] = publish_release(GitHubReleaseClient(token), artifacts, args.release_tag)
    else:
        report['github'] = 'NOT_ATTEMPTED_DRY_RUN'
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
