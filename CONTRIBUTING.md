# Contributing

## Setup

```bash
git clone <repo-url>
cd automated-ml-pipeline
make dev
```

## Development Workflow

1. Create a feature branch from `main`
2. Write code + tests
3. Run `make lint` and `make test`
4. Open a pull request

## Code Standards

- Python 3.11+
- Format with `ruff format`
- Lint with `ruff check`
- Minimum 80% test coverage
- Type hints on all public functions

## Testing

```bash
make test-unit       # Unit tests only
make test-integration # Integration tests only
make coverage        # Full coverage report
```

## Commit Messages

Use conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `ci:`.
