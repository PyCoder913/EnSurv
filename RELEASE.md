# Publishing EnSurv

Run these commands in PowerShell from the repository root. Use Python 3.12 for the release environment if available so your local documentation build uses the same Python version as Read the Docs.

## 1. Review the release metadata and staged files

- Confirm `name = "ensurv"` and `version = "0.1.0"` in `pyproject.toml`. Check both project URLs before uploading. If the name is unavailable, change the distribution name in `pyproject.toml`, `docs/conf.py`, and the installation examples, and use the new name in the commands below. The import name `ensurv` can stay the same. [PyPI project lookup](https://pypi.org/project/ensurv/) · [TestPyPI project lookup](https://test.pypi.org/project/ensurv/)
- Confirm the release license and author list with the rights holders. Add the approved license text as `LICENSE`, set `license` to its SPDX expression and `license-files = ["LICENSE"]` under `[project]`, and add `authors` if desired. Confirm permission to redistribute the notebooks and any results they contain. The current metadata leaves licensing and authorship for you to specify. [PyPA project metadata](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
- Inspect the index with `git status --short` and `git diff --cached --stat`. Stage any metadata updates you make. The repository already has a local `main` branch ready for its first commit.

## 2. Build the docs locally

Create a release environment and install the project with its docs and workflow extras. If `python` selects a different interpreter, use the full path to Python 3.12 in the first line.

```powershell
python -m venv .venv-release
$ReleasePython = ".\.venv-release\Scripts\python.exe"
& $ReleasePython -m pip install --upgrade pip build twine
& $ReleasePython -m pip install --index-url https://download.pytorch.org/whl/cpu torch
& $ReleasePython -m pip install ".[docs,workflow]"
& $ReleasePython -m sphinx -W -b html docs docs/_build/html
Start-Process (Resolve-Path docs/_build/html/index.html)
```

Check the landing page, Quickstart, API reference, and notebook downloads. The CPU wheel is sufficient for the documentation environment. [PyTorch installer](https://pytorch.org/get-started/locally/)

## 3. Push the source and connect Read the Docs

1. Create a public, empty repository on GitHub. GitLab or Bitbucket also work with the corresponding Read the Docs integration. Use the actual repository URL in the command below.

   ```powershell
   git diff --cached --check -- .gitignore .readthedocs.yaml MANIFEST.in pyproject.toml README.md RELEASE.md docs
   git commit -m "Prepare EnSurv package and documentation"
   git remote add origin https://github.com/YOUR-ACCOUNT/ensurv.git
   git push -u origin main
   ```

2. Sign in to Read the Docs Community, connect your Git provider, and choose **Add project**. Select the EnSurv repository and complete the import. The integration sets up automatic builds from Git pushes. For a manual import, configure the provider webhook using the project’s Integrations settings. [Read the Docs import tutorial](https://docs.readthedocs.com/platform/stable/tutorial/index.html)
3. Open the project’s **Builds** tab and wait for `latest` from `main` to succeed. Read the Docs reads `.readthedocs.yaml` from the repository root, installs the CPU build of PyTorch and the package extras, and treats Sphinx warnings as build failures. Use the build log if a step fails. [Read the Docs configuration reference](https://docs.readthedocs.com/platform/stable/config-file/v2.html)
4. Open **View docs** and copy the working URL. Add the source and documentation URLs to `pyproject.toml` under `[project.urls]`, using `Repository` and `Documentation` as keys. You can also add the docs link to `README.md`. Commit and push those edits before building release files.

   ```powershell
   git add -- pyproject.toml README.md
   git commit -m "Add repository and documentation links"
   git push
   ```

## 4. Build and check the release artifacts

Use a new output folder for each build attempt and upload its exact wheel and source archive. Run the following in the PowerShell session from step 2.

```powershell
$ReleaseVersion = "0.1.0"
$ReleaseOutput = "dist/$ReleaseVersion"
if (git status --porcelain) { throw "Commit the release changes before building." }
if (Test-Path $ReleaseOutput) { throw "Choose a fresh release output folder." }

& $ReleasePython -m build --outdir $ReleaseOutput
if ($LASTEXITCODE -ne 0) { throw "Package build failed." }

$Wheel = "$ReleaseOutput/ensurv-$ReleaseVersion-py3-none-any.whl"
$Sdist = "$ReleaseOutput/ensurv-$ReleaseVersion.tar.gz"
& $ReleasePython -m twine check --strict $Wheel $Sdist
if ($LASTEXITCODE -ne 0) { throw "Distribution metadata check failed." }
```

Install the wheel in a separate environment and run the README’s fit-and-predict snippet with `$SmokePython`. The wheel provides the Python library. The source archive also includes the documentation and example notebooks.

```powershell
python -m venv .venv-smoke
$SmokePython = ".\.venv-smoke\Scripts\python.exe"
& $SmokePython -m pip install --index-url https://download.pytorch.org/whl/cpu torch
& $SmokePython -m pip install $Wheel
& $SmokePython -m pip check
& $SmokePython -c "from importlib.metadata import version; from ensurv import EnSurv; print(version('ensurv'))"
```

## 5. Rehearse on TestPyPI

Create a TestPyPI account, enable two-factor authentication, and create an API token. For the first upload, use the account-wide scope if the project does not yet exist. Run the upload and enter `__token__` when Twine asks for the username and the full `pypi-…` token when it asks for the password. [PyPA TestPyPI guide](https://packaging.python.org/en/latest/guides/using-testpypi/)

```powershell
& $ReleasePython -m twine upload --repository testpypi $Wheel $Sdist
```

Check the TestPyPI project page. Reinstall the uploaded wheel in the smoke environment, which already has dependencies from production PyPI, and check the version and import again.

```powershell
& $SmokePython -m pip install --force-reinstall --index-url https://test.pypi.org/simple/ --no-deps --only-binary=:all: "ensurv==$ReleaseVersion"
& $SmokePython -m pip check
& $SmokePython -c "from importlib.metadata import version; from ensurv import EnSurv; print(version('ensurv'))"
```

## 6. Upload to PyPI and tag the released source

Create a separate production PyPI account and API token with two-factor authentication enabled. Upload the same artifacts you tested. Use `__token__` as the username and the production token as the password. After the first successful upload, you can replace the account-wide token with a token scoped to this project. [PyPA publishing tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/)

```powershell
& $ReleasePython -m twine upload --repository pypi $Wheel $Sdist
& $SmokePython -m pip install --force-reinstall --index-url https://pypi.org/simple/ --no-deps --only-binary=:all: "ensurv==$ReleaseVersion"
& $SmokePython -m pip check
& $SmokePython -c "from importlib.metadata import version; from ensurv import EnSurv; print(version('ensurv'))"
```

Check the PyPI project page for the rendered README and both download files. Once the upload and install checks pass, tag the exact commit you built from. The following commands assume you are still on that commit.

```powershell
git tag -a "v$ReleaseVersion" -m "EnSurv $ReleaseVersion"
git push origin "v$ReleaseVersion"
```

In Read the Docs, check the `stable` build and enable the version for the release tag if you want it to appear in the version menu. Open the hosted `latest` and `stable` pages and verify which version each serves. You can set `stable` as the project’s default version. [Read the Docs version documentation](https://docs.readthedocs.com/platform/stable/versions.html)

For later releases, update the version in `pyproject.toml`, commit and push the changes, rebuild into a fresh folder, and repeat the upload and tagging steps. PyPI distribution filenames cannot be reused. If you automate publishing later, use [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/).
