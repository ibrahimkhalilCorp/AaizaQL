# T1.2 — Coverage badge setup instructions
#
# Add the two lines below to the top of README.md,
# replacing YOUR_GITHUB_ORG/aaizaql with the real repo slug.
# ─────────────────────────────────────────────────────────────────────────────

## Badge markdown (paste into README.md)

[![Coverage](https://codecov.io/gh/YOUR_GITHUB_ORG/aaizaql/branch/main/graph/badge.svg)](https://codecov.io/gh/YOUR_GITHUB_ORG/aaizaql)
[![CI](https://github.com/YOUR_GITHUB_ORG/aaizaql/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_GITHUB_ORG/aaizaql/actions/workflows/ci.yml)

## One-time Codecov setup

1. Go to https://codecov.io and sign in with GitHub.
2. Add the aaizaql repository (it will appear automatically for public repos).
3. For private repos: copy the upload token from Codecov and add it as a
   GitHub Actions secret named CODECOV_TOKEN, then add this to ci.yml:
     env:
       CODECOV_TOKEN: ${{ secrets.CODECOV_TOKEN }}
4. Push any commit — CI will upload coverage-unit.xml and the badge will
   appear within ~60 seconds of the run completing.

## Viewing the HTML report locally

pytest tests/unit/ \
  --cov=src/aaizaql \
  --cov-config=.coveragerc-unit \
  --cov-report=html:htmlcov/unit

open htmlcov/unit/index.html   # macOS
start htmlcov\unit\index.html  # Windows
