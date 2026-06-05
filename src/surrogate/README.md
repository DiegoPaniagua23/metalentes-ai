# src/surrogate/

Entrenamiento y evaluacion del surrogate multi-output.

## Modelo

El surrogate recibe una geometria de 102 dimensiones y predice:

- Amplitud focal `|E|(y)`.
- Fase focal `Phi(y)`.
- `R2_pred`, estimacion directa del criterio fisico.

## Uso

```bash
python src/surrogate/train_field_surrogate.py \
  --epochs 200 --batch_size 64 --hidden_dims 512,512,512 \
  --dropout 0.1 --lr 1e-3 \
  --weight_amp 1.0 --weight_phase 1.0 --weight_r2 10.0

python src/surrogate/evaluate_field_surrogate.py
```

## Entradas

- `data/dataset/dataset_train.npz`
- `data/dataset/dataset_val.npz`
- `data/dataset/dataset_test.npz`

## Salidas

- `results/surrogate/model.pt`
- `results/surrogate/training_log.json`
- `results/surrogate/eval_metrics.json`
- `results/surrogate/config.json`

El modelo y las metricas son regenerables. Solo `config.json` queda versionado como referencia ligera.
