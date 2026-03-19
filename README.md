# INTRO2MODELSNEU

## Requirements

Install pixi: https://pixi.prefix.dev/latest/installation/

## Quickstart

If you have the `Manifest.toml` (which you do if you cloned this repo), this is all you need:

```bash
git clone git@github.com:dksasaki/intro2modelsNEU.git
cd intro2modelsNEU
pixi install
pixi run instantiate-kernel
pixi run lab
```

- `pixi install` — installs Julia, JupyterLab, and conda from `pixi.lock`
- `pixi run instantiate-kernel` — installs Julia packages (IJulia, Oceananigans, CairoMakie) from `Manifest.toml`
- `pixi run lab` — starts JupyterLab in your browser

## First time setup (maintainers only)

If you are starting from scratch without a `Manifest.toml`:

```bash
pixi install
pixi run install-kernel
pixi run install-oceananigans
pixi run install-cairomakie
```

Then commit `Project.toml` and `Manifest.toml` to the repository.

## Reproducibility

| File | Purpose |
|------|---------|
| `pixi.toml` | defines system-level dependencies (Julia, JupyterLab, conda) |
| `pixi.lock` | pins exact package versions |
| `Project.toml` | defines Julia dependencies |
| `Manifest.toml` | pins exact Julia package versions |

Do not delete any of these files. The `.pixi/` directory is local and can be safely removed and recreated with `pixi install`.