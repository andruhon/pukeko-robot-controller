# Contributing

Thanks for your interest in improving Pukeko Robot Controller! This is an
educational project — please read the safety disclaimer in
[SECURITY.md](./SECURITY.md) before running it against real hardware.

By participating, you agree to abide by our
[Code of Conduct](./CODE_OF_CONDUCT.md).

## Getting started

1. Fork and clone the repo.
2. `npm install`
3. See [AGENTS.md](./AGENTS.md) for the architecture, how to run the app, and
   how to smoke-test it (the README's "Running…" sections cover the commands).

## Before opening a PR

- `npm run type-check` — must pass.
- `npm test` — must pass.
- New code is formatted with Prettier (`.prettierrc.json`); match the style of
  the surrounding code.
- Keep PRs focused, and describe what changed and why.

## Local development registry (Verdaccio)

This repo depends on `@galvanized-pukeko/*` and `@gaunt-sloth/*`, which are
published to the public npm registry. When developing against **unpublished**
versions of those packages, route their scopes to a local
[Verdaccio](https://verdaccio.org) registry at `http://localhost:4873`.

### Consuming unpublished versions

Create a gitignored `.npmrc` at the repo root:

```
@gaunt-sloth:registry=http://localhost:4873
@galvanized-pukeko:registry=http://localhost:4873
```

Then `npm install` pulls those scopes from Verdaccio and everything else from
the public registry. To consume only the published versions, delete `.npmrc`
before installing.

Verdaccio container and auth setup is documented in
[gaunt-sloth-assistant/CONTRIBUTING.md](https://github.com/Galvanized-Pukeko/gaunt-sloth-assistant/blob/main/CONTRIBUTING.md#local-development-registry-optional).

### Publishing a local build to Verdaccio

To test a change to `@galvanized-pukeko/vue-ui` (its source lives in its own
repo) inside this project before cutting a real npm release:

```sh
# in the vue-ui package's repo
npm version patch --no-git-tag-version -w @galvanized-pukeko/vue-ui
npm run build -w @galvanized-pukeko/vue-ui
npm publish -w @galvanized-pukeko/vue-ui --registry http://localhost:4873

# back in this repo
npm install @galvanized-pukeko/vue-ui@<version> --registry http://localhost:4873
```

> ⚠️ That install rewrites `package-lock.json` to resolve the package from
> `localhost:4873`. **Don't commit that lockfile** — cut a real release to the
> public registry first, then re-resolve the lockfile against it.

## Reporting bugs and security issues

For functional bugs, open an issue. For anything security- or safety-related,
follow [SECURITY.md](./SECURITY.md).
