# Coverage Badge Setup (T1.2)

Add the badges below to the top of `README.md`, replacing `YOUR_GITHUB_ORG`
with the real repo owner (e.g. `ibrahimkhalil`).

## Badges

```markdown
[![Coverage](https://codecov.io/gh/YOUR_GITHUB_ORG/aaizaql/branch/main/graph/badge.svg)](https://codecov.io/gh/YOUR_GITHUB_ORG/aaizaql)
[![CI](https://github.com/YOUR_GITHUB_ORG/aaizaql/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_GITHUB_ORG/aaizaql/actions/workflows/ci.yml)
```

## One-time Codecov setup

1. Go to <https://codecov.io> and sign in with GitHub.
2. Add the `aaizaql` repository (appears automatically for public repos).
3. **Private repo only:** copy the upload token from Codecov, add it as a
   GitHub Actions secret named `CODECOV_TOKEN`, then add this to `ci.yml`
   under the Codecov upload step:
   ```yaml
   env:
     CODECOV_TOKEN: ${{ secrets.CODECOV_TOKEN }}
   ```
4. Push any commit — CI uploads `coverage-unit.xml` and the badge updates
   within ~60 seconds of the run completing.

## Viewing the HTML report locally

```bash
pytest tests/unit/ \
  --cov=src/aaizaql \
  --cov-config=.coveragerc-unit \
  --cov-report=html:htmlcov/unit

open htmlcov/unit/index.html    # macOS
start htmlcov\unit\index.html   # Windows
xdg-open htmlcov/unit/index.html  # Linux
```
