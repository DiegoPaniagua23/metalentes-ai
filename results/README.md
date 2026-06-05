# results/

Resultados del pipeline.

## Estructura

```text
results/
|-- final/            # Entregables finales versionados
|-- surrogate/        # Modelo y metricas regenerables
`-- inverse_design/   # Candidatos y convergencia GA regenerables
```

## Versionado

Se versiona:

- `results/final/candidates_ranked.csv`
- `results/final/basin_summary.json`
- Configuraciones ligeras necesarias para interpretar ejecuciones.

No se versiona:

- Modelos `.pt`.
- Logs de entrenamiento.
- Metricas regenerables.
- `top_candidates*.csv` intermedios.
- `ga_convergence.json`.

## Resultado Principal

`results/final/candidates_ranked.csv` contiene la geometria recomendada `hc627_036` con `R2_real = 0.9256`.
