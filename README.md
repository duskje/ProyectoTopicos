# Proyecto semestral "Tópicos en Manejo de Grandes Volúmenes de Datos"
## Requisitos
Para ejecutar este proyecto es necesario tener `poetry` y al menos Python 3.11.

## Instalación
Creamos un entorno virtual para el proyecto con poetry.

```bash
git clone https://github.com/duskje/ProyectoTopicos
cd ProyectoTopicos
poetry install
```

## Simulación
Para ejecutar una simulación es necesario ejecutar:

```bash
poetry run simulation
```

para la comparación entre PCSA y Sketch-Flip-Merge, se puede ejecutar:

```bash
poetry run examples
```

## Patentes
El archivo analysis.py incluye cálculos sobre patentes para sketches de distintos parámetros.

Por ejemplo:

```python
from analysis import find_plates_until_n_leading_zeros_hll

find_plates_until_n_leading_zeros_hll(n=3, p=14)
```

nos dará las patentes chilenas con hasta 3 leading zeros para un sketch HyperLogLog p=14.

## Tests
Para ejecutar los tests para los estimadores de cardinalidad

```bash
poetry run pytest test_cardinality_estimation
```

## Referencias
[1] Damien Desfontaines, Andreas Lochbihler, and David Basin. Cardinality Estimators do
not Preserve Privacy. Proceedings on Privacy Enhancing Technologies, 2019(2):26–46, 4 2019.

[2] Cynthia Dwork. Differential privacy. 1 2006.

[3] Jonathan Hehir, Daniel Shu Wei Ting, and Graham Cormode. Sketch-Flip-Merge: Mer-
geable sketches for private distinct counting. arXiv (Cornell University), 2 2023.