# Doehlert Suite 0.1.0

**English is the default application language. Spanish is available from the application's _Language_ menu.** Spanish documentation is provided in [`README.es.md`](README.es.md) and `docs/es/`.

Doehlert Suite is an open-source Python desktop application for generating and analyzing three-factor Doehlert experimental designs. It fits quadratic response-surface models, reports ANOVA and diagnostics, performs constrained optimization, supports multiple responses, and includes an integrated audit utility.

> **Release status:** 0.1.0 is the first public release. Treat results as analytical support, not as a substitute for experimental validation, domain expertise, or independent statistical review.

## Features

- Generate and visualize three-factor Doehlert designs.
- Import experimental data from CSV files.
- Fit a full second-order (quadratic) response-surface model.
- Review ANOVA, coefficient inference, lack-of-fit, and residual diagnostics.
- Search for constrained maxima and minima within the supported experimental domain.
- Import and analyze multiple responses from a wide-format CSV.
- Run the integrated audit and export audit evidence.
- Switch the interface between English (default) and Spanish.
- Practice with synthetic datasets and validation cases in `Examples/`.

## Requirements

- Python 3.10 or newer.
- Tkinter. On some Linux distributions install it with your system package manager (for Debian/Ubuntu: `sudo apt install python3-tk`).
- NumPy, Matplotlib, and SciPy (see `requirements.txt`).
- `openpyxl` and `statsmodels` are optional dependencies for supported supplementary features.

## Installation

```bash
git clone https://github.com/YOUR-USERNAME/Doehlert-Suite.git
cd Doehlert-Suite
python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install and launch:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

## Quick start

1. Open **1. Design**, set the names and physical limits of the three factors, and generate the design matrix.
2. Export the design CSV and enter your measured response values in the response column.
3. Use **File → Import Experimental Data (CSV)** to load the completed file.
4. Review the model, optimum, ANOVA, coefficients, and diagnostics in the analysis tabs.
5. Use the audit and multiple-response tab when needed.
6. Export reports and tables. New exports are grouped under `Exports/YYYY-MM-DD/`.

The files in `Examples/` are synthetic demonstration or validation datasets. They are not real experimental measurements.

## Language selection

The interface starts in English. Choose **Language → Español** to switch to Spanish. The application asks to restart so all windows and labels can be rebuilt consistently. The selection is stored in the current user's home directory, outside the repository. Choose **Language → English** to return to English.

## Documentation

- [English technical and user manual (TXT)](docs/en/TECHNICAL_MANUAL.txt)
- [English technical documentation (PDF)](docs/Technical_Documentation_EN.pdf)
- [English quick start](docs/en/QUICK_START.txt)
- [Spanish technical and user manual (TXT)](docs/es/TECHNICAL_MANUAL.txt)
- [Spanish technical documentation (PDF)](docs/Documentacion_Tecnica_ES.pdf)
- [Spanish quick start](docs/es/QUICK_START.txt)
- [Documentation index](docs/README.md)

## Automated tests

From the repository root:

```bash
python -m unittest discover -s auditor/tests -v
```

The tests cover selected domain, optimization, multiple-response, and publication-comparison behavior. Passing tests do not prove that every possible dataset or workflow is error-free.

## Repository layout

```text
Doehlert-Suite/
├── main.py                     # Desktop user interface
├── i18n.py                     # English/Spanish localization
├── doehlert_engine.py          # Statistical engine and model fitting
├── doehlert_domain.py          # Experimental-domain geometry
├── doehlert_optimization.py    # Constrained optimization
├── doehlert_responses.py       # Multiple-response CSV handling
├── doehlert_publication.py     # Published-result comparison
├── auditor/                    # Integrated audit module and tests
├── Examples/                   # Demonstration and validation CSV files
├── docs/en/                    # English manuals
├── docs/es/                    # Spanish manuals
├── requirements.txt
├── LICENSE                     # GPL-3.0-only
└── CHANGELOG.md
```

`Exports/` is created when the application runs and is intentionally not versioned. Temporary files, caches, and local user settings are excluded from Git.

## Scope and limitations

- The current design implementation supports three factors.
- Model quality depends on the quantity, quality, and structure of the input data.
- Numerical optimization is conditional on the fitted model and configured constraints.
- Synthetic examples are not a substitute for validating the workflow with your own experimental data.
- Version 0.1.0 is an initial public release. When reporting an issue, include reproduction steps, operating system, Python version, and dependency versions. Do not attach confidential data.

## License

Doehlert Suite is distributed under the **GNU General Public License v3.0 only (GPL-3.0-only)**. You may use, study, modify, and redistribute it under the terms of that license. Covered redistributions and derivative works must comply with the GPL, including its source-code requirements. See [`LICENSE`](LICENSE). Third-party dependencies retain their own licenses.

## Contributing

Contributions are welcome. Open an issue describing the problem or submit a pull request with a concise explanation and tests where appropriate. Please keep mathematical formulas, statistical assumptions, and domain constraints documented.
