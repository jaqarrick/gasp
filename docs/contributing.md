# Contributing

## Dev Containers

This project includes a [dev container](https://containers.dev/) configuration for a consistent development environment.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [VS Code](https://code.visualstudio.com/) with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### Getting Started

1. Clone the repo:
```bash
   git clone --recurse-submodules git@github.com:jaqarrick/gasp.git
```
2. Open in VS Code
3. When prompted, click **"Reopen in Container"** (or run `Dev Containers: Reopen in Container` from the command palette)
4. The container will build with all dependencies (Python, CUDA Toolkit)

### Without VS Code

```bash
docker build -t gasp-dev .
docker run -it --rm -v $(pwd):/app gasp-dev /bin/bash
```

## Commit Conventions & Hooks

This repo uses [pre-commit](https://pre-commit.com/) to enforce formatting, linting, and commit message style. `make install` sets up the hooks; to install them manually, run `pre-commit install`.

On every commit:

- **pre-commit**: `ruff check --fix` and `ruff format` run on staged files. If they modify anything, the commit stops — `git add` the changes and commit again.
- **commit-msg**: the message must follow [Conventional Commits](https://www.conventionalcommits.org/):

  ```
  type(optional scope): subject
  ```

  Allowed types: `feat`, `fix`, `chore`, `docs`, `refactor`, `test`, `ci`, `build`, `perf`, `style`, `revert`.

To run every hook against the whole repo: `make pre-commit`.
