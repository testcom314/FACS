# Documentation Index

This project is documented as a measurement and analysis system rather than as an “emotion detector.”

| File | Purpose |
|---|---|
| `GETTING_STARTED.md` | Install and run a first analysis |
| `THEORY_AND_METHODS.md` | Explain the measurement model and interpretation limits |
| `TECHNICAL_DOCUMENTATION.md` | Explain the implementation and processing pipeline |
| `DATA_FORMAT.md` | Explain exported fields and session files |
| `CONFIGURATION.md` | Explain settings and how changing them affects analysis |
| `VALIDATION.md` | Explain internal and external validation |
| `DEVELOPMENT_HISTORY.md` | Record how the system evolved and why major changes were made |
| `RESEARCH_NOTES.md` | Record the research used to guide the design |
| `TROUBLESHOOTING.md` | Common installation and runtime problems |
| `API_REFERENCE.md` | Programmatic entry points and classes |

## Recommended reading order

For a first-time user:

```text
GETTING_STARTED
    -> DATA_FORMAT
    -> THEORY_AND_METHODS
```

For someone modifying the software:

```text
TECHNICAL_DOCUMENTATION
    -> CONFIGURATION
    -> VALIDATION
    -> API_REFERENCE
```

For understanding why the project is designed this way:

```text
THEORY_AND_METHODS
    -> RESEARCH_NOTES
    -> DEVELOPMENT_HISTORY
```
