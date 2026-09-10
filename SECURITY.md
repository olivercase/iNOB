# Security

iNOB is research software that runs locally: it reads mesh files and a YAML
config and writes arrays and figures. It has no accounts, no network calls
of its own, and stores no personal data. The browser GUI binds to localhost
and is not designed to be exposed to a network.

If you find a vulnerability — a config value that reaches the filesystem or
a shell outside `outputs/`, an unsafe deserialisation, a dependency advisory
that affects a code path here — email oliver@fizzymilk.com rather than
opening a public issue. Expect an acknowledgement within a week.

Automated checks: gitleaks runs in the pre-commit hooks and in CI, Dependabot
watches the Python, npm and GitHub-Actions dependencies, and CI actions are
pinned to commit SHAs.
