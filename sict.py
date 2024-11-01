#!/usr/bin/env python3

from enum import Enum
import pickle
from dsp import Optional
from dspy.datasets import DataLoader
import pandas as pd
import dspy
import pydantic

from dspy_test import setup_phoenix

df = pd.read_csv("20240724_bs2.csv")

# df.sample(10).to_csv("sict.csv")
df_danos: pd.DataFrame = df.loc[
    (df["dano_moral"] == "SIM") | (df["dano_material"] == "SIM")
].sample(5)

df = pd.concat([df_danos, df.sample(5)])
# print(df)

lm = dspy.OllamaLocal(model="mistral-nemo")


class Boolean(Enum):
    yes = "SIM"
    no = "NÃO"


class Resposta(pydantic.BaseModel):
    dano_moral: Optional[Boolean]
    valor_dano_moral: Optional[str]
    dano_material: Optional[Boolean]
    valor_dano_material: Optional[str]


class MovimentacaoClassificada(dspy.Signature):
    """Sua tarefa é extrair os dados da movimentação abaixo seguindo o esquema dos campos. Me fale há presença de dano moral ou material e seus respectivos valores. Para identificar valores, utilize contextos como 'no valor de R$ {valor numérico}'"""

    movimentacao = dspy.InputField()
    reposta: Resposta = dspy.OutputField()


class Classificador(dspy.Module):
    def __init__(self, signature):
        super().__init__()
        self.saida = dspy.functional.TypedPredictor(
            signature, max_retries=6, explain_errors=True
        )

    def forward(self, movimentacao: dspy.Example):
        data = self.saida(movimentacao=movimentacao.movimentacao)
        return data


class SaveData:
    def __init__(self, movimentacao, pred) -> None:
        self.movimentacao = movimentacao
        self.pred = pred


if __name__ == "__main__":
    setup_phoenix()
    modelo = Classificador(MovimentacaoClassificada)

    splits = DataLoader().from_pandas(
        df, ["movimentacao"], input_keys=("movimentacao",)
    )
    result = open("resultados.pkl", "wb")
    dt = []
    for el in splits:
        pred = modelo(movimentacao=el)
        saveData = SaveData(el.movimentacao, pred)
        dt.append(saveData)
