# Releasing `lumina-sdk`

Releases are tag-driven and published via GitHub Actions using **PyPI Trusted
Publishing** (OIDC). No long-lived API tokens are stored anywhere.

## One-time setup (already done; documented for future maintainers)

1. **TestPyPI Pending Publisher** — https://test.pypi.org/manage/account/publishing/
   - PyPI Project Name: `lumina-sdk`
   - Owner: `cshjin`
   - Repository name: `lumina-sdk`
   - Workflow name: `release.yml`
   - Environment name: `testpypi`
2. **PyPI Pending Publisher** — https://pypi.org/manage/account/publishing/ — same
   fields except environment name `pypi`.
3. **GitHub Environments** — Settings -> Environments:
   - Create `testpypi` (no required reviewers).
   - Create `pypi` and add a required reviewer so production publishes need a
     manual click.
4. **Squat-protect** the `lumina-sdk` name on both indexes by completing the
   first `rc` cycle (below).

## Cutting a release

### 1. Bump version

Edit `pyproject.toml`:

```toml
version = "X.Y.Z"
```

`lumina/__init__.py` reads the version from package metadata, so no other file
needs editing. Confirm:

```shell
python -c "import tomllib, pathlib; print(tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']['version'])"
```

### 2. Update CHANGELOG

Move the `[Unreleased]` content into a new `[X.Y.Z] - YYYY-MM-DD` section.
Leave `[Unreleased]` empty for the next cycle.

### 3. Open and merge a release PR

```shell
git checkout -b release/vX.Y.Z
git commit -am "release: vX.Y.Z"
gh pr create --base main --title "release: vX.Y.Z" --body "See CHANGELOG.md"
# review and merge
```

### 4. Tag a release candidate -> TestPyPI

```shell
git checkout main && git pull
git tag vX.Y.Zrc1
git push origin vX.Y.Zrc1
```

The `release.yml` workflow runs:

- `build` — builds sdist + wheel, runs `twine check`.
- `publish-testpypi` — fires because the tag contains `rc`.

Wait for the workflow to succeed (~2 min) plus a few minutes of TestPyPI mirror
lag, then smoke-test in a fresh venv:

```shell
python -m venv /tmp/lumina-rc && source /tmp/lumina-rc/bin/activate
pip install -i https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            lumina-sdk==X.Y.Zrc1
python -c "import lumina; print(lumina.__version__)"
```

`--extra-index-url` is required because TestPyPI doesn't host torch/numpy/etc.

If the smoke test fails, fix forward with `rc2`. **Never reuse `rc1`** — PyPI
versions are immutable.

### 5. Tag the real release -> PyPI

Once an `rc` smoke-tests cleanly:

```shell
git tag vX.Y.Z
git push origin vX.Y.Z
```

The `publish-pypi` job fires (no `rc`/`test` in the tag name). It pauses on the
`pypi` environment until a reviewer approves it in the GitHub Actions UI.
After approval, the wheel + sdist publish to PyPI.

Verify:

```shell
pip install lumina-sdk==X.Y.Z
python -c "import lumina; print(lumina.__version__)"
```

Confirm https://pypi.org/project/lumina-sdk/ renders the README and license.

## Tag naming conventions

| Tag | Lands on | Triggered by |
|---|---|---|
| `vX.Y.Zrc1`, `vX.Y.Zrc2`, ... | TestPyPI | `contains(ref_name, 'rc')` |
| `vX.Y.Z-test`, `vX.Y.Z-test2` | TestPyPI | `contains(ref_name, 'test')` |
| `vX.Y.Z` | PyPI | otherwise |

Stick to PEP 440: prefer `0.1.0rc1` over `0.1.0-rc.1`.

## Hotfix releases

For an urgent fix to a published version `X.Y.Z`:

1. Branch from the release tag: `git checkout -b hotfix/X.Y.Z+1 vX.Y.Z`
2. Apply the fix, bump version to `X.Y.Z+1` (e.g., `0.1.1`), update CHANGELOG.
3. Merge to `main` via PR.
4. Tag `vX.Y.Z+1` from `main` and push.

`X.Y.Z.post1` is also valid for pure-metadata fixes (no code changes), but
prefer a real patch bump unless you specifically want post-release semantics.

## Things that will fail

- **Re-tagging the same version after a failed publish.** PyPI never accepts a
  re-upload of an already-published version, even after it's been "deleted".
  Bump the patch (`0.1.0 -> 0.1.1`) or use a `.postN` suffix.
- **Trusted publishing 403.** GitHub `environment:` field must match the PyPI
  Pending Publisher config exactly. Check both names character-for-character.
- **TestPyPI `pip install` 404 right after a successful publish.** Mirror lag.
  Wait 1-3 minutes.
- **Resolver failure on torch.** If you tighten `dependencies` in
  `pyproject.toml` to a torch version not yet on PyPI for the user's Python,
  install fails before download. Verify `pip index versions torch` covers all
  classifier-claimed Python versions before tagging.
