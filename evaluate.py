#!/usr/bin/env python3

import dspy_test
from dspy.evaluate import Evaluate
from dspy.datasets import DataLoader
import dspy

dspy_test.setup_phoenix()

llm = dspy.OllamaLocal(model="mistral-nemo")
dspy.configure(lm=llm)
modelo = dspy_test.Classificador(dspy_test.MovimentacaoClassificada)

modelo.load("classificador.json")

dl = DataLoader()
dataset = dl.from_csv(
    "./20240724_bs2_com_partes.csv",
    fields=list(
        (
            "dano_moral",
            "valor_dano_moral",
            "dano_material",
            "valor_dano_material",
            "movimentacao",
        )
    ),
    input_keys=("movimentacao",),
)
print(dataset[0].labels())
evaluate = Evaluate(
    devset=dataset[:300],
    metric=dspy_test.metrica_extracao,
    display_progress=True,
    num_threads=6,
)

evaluate(modelo)
