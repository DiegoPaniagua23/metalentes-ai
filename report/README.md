# report/

Fuentes LaTeX del reporte tecnico.

## Estructura

```text
report/
|-- Reporte.tex
|-- Preambulo.tex
|-- Portada/
|-- Capitulos/
|-- Figuras/
`-- Bibliografia/
```

## Compilar

Desde `report/`:

```bash
latexmk -pdf -outdir=out Reporte.tex
```

El PDF y auxiliares quedan en `report/out/`, que no se versiona.

## Figuras

Las figuras curadas del reporte se versionan en `report/Figuras/`. El script `report/Figuras/generate_figures.py` documenta como regenerar las figuras principales desde los resultados disponibles.

## Nota

El reporte es un entregable fuente. Los artefactos compilados se regeneran localmente y no deben agregarse al repo.
