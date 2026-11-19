# Contributing

Thanks for your interest in contributing!

## Development setup

```bash
git clone <this repo>
cd <repo>
bundle install         # Ruby projects
# or
npm ci                 # JS projects
```

## Workflow

1. Fork → branch from `main`
2. Make changes with tests
3. Run `bundle exec rspec` (Ruby) or `npm test` (JS) locally
4. Run `bundle exec standardrb` (Ruby) or `npm run lint` (JS)
5. Open a PR with a clear description

## Code style

- Ruby: enforced by [StandardRB](https://github.com/standardrb/standard)
- JavaScript: enforced by ESLint + Prettier
- Python: enforced by ruff

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new transliteration system
fix: correct off-by-one in CALT lookup
chore: bump dependencies
docs: clarify README
```

## Releases

Maintainers tag releases following semver. CI publishes on tag push.

## Ribose Contributor License Agreement

Before your first pull request can be merged in any Ribose GitHub
organization, you need to be covered by the Ribose Contributor License
Agreement (CLA). Sign once; one signature covers every Ribose
organization.

- **Individual contributors** — sign the Individual CLA at
  <https://cla.ribose.com/icla>. The `cla/ribose` check on your pull
  request reports the state, and you can also sign from the pull request
  by commenting `@ribose-cla agree` and completing the one-click link
  the App replies with.
- **Corporate contributors** — if your employer has an executed
  Organization CLA, the employer's Point of Contact adds your GitHub
  handle to its Schedule A at
  <https://cla.ribose.com/organizations>, or you can request coverage
  there yourself. No individual signature is then needed.
- **The full guide** — the signing flows and the agreement text are at
  <https://github.com/riboseinc/cla/blob/main/docs/SIGNING.md>.
