#!/usr/bin/env python3

from enum import Enum
from dsp import Optional
import dspy
from dspy.datasets import DataLoader
import pandas as pd
import pydantic
import sys
from dspy_test import setup_phoenix

dl = DataLoader()

df = pd.read_csv("./Fundo Ativos - Leads - LEADS.csv").sample(50)

print(df.head(4))


class Boolean(Enum):
    yes = "SIM"
    no = "NÃO"


class Resposta(pydantic.BaseModel):
    dano_moral: Boolean
    valor_dano_moral: Optional[str]
    dano_material: Boolean
    valor_dano_material: Optional[str]


class MovimentacaoClassificada(dspy.Signature):
    """Sua tarefa é extrair os dados da movimentação abaixo seguindo o esquema dos campos. Me fale há presença de dano moral ou material e seus respectivos valores. Para identificar valores, utilize contextos como 'no valor de R$ {valor numérico}'"""

    movimentacao = dspy.InputField()
    resposta: Resposta = dspy.OutputField()


class Classificador(dspy.Module):
    def __init__(self, signature):
        super().__init__()
        self.saida = dspy.functional.TypedPredictor(
            signature, max_retries=6, explain_errors=True
        )

    def forward(self, movimentacao: dspy.Example):
        data = self.saida(movimentacao=movimentacao.movimentacao)
        return data


lm = dspy.OllamaLocal(model="mistral-nemo")
if __name__ == "__main__":
    setup_phoenix()
    modelo = Classificador(MovimentacaoClassificada)

    splits = DataLoader().from_pandas(
        df, ["movimentacao", "numero_processo"], input_keys=("movimentacao",)
    )
    dt = []
    for el in splits:
        try:
            pred = modelo(movimentacao=el)
            dt.append(
                dict(
                    pred.resposta,
                    movimentacao=el.movimentacao,
                    num_process=el.numero_processo,
                )
            )
        except KeyboardInterrupt:
            sys.exit(0)
        except:
            print("erro")
    output = pd.DataFrame(dt)
    output.to_csv("saida.csv")
