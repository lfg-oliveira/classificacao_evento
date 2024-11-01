#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy.datasets import DataLoader
import pandas as pd
from enum import Enum
import pydantic

from openinference.instrumentation.dspy import DSPyInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
span_otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
tracer_provider.add_span_processor(
    SimpleSpanProcessor(span_exporter=span_otlp_exporter)
)

trace_api.set_tracer_provider(tracer_provider=tracer_provider)
DSPyInstrumentor().instrument()


llm = dspy.OllamaLocal("command-r")


class Tipo(Enum):
    acordao = "Acórdão"
    embargo = "Embargo"
    sentenca = "Sentenca"


class TipoSentenca(Enum):
    extincao = "Extinção"
    parcial = "Procedência Parcial"
    total = "Procedência Total"
    improcedente = "Improcedente"


class GeradorModel(pydantic.BaseModel):
    tipo: Tipo
    sentenca: TipoSentenca


class GeradorSignature(dspy.Signature):
    """Sua tarefa é gerar dados sintéticos com base na entrada, porém deve ser distinto desta."""

    texto_base: str = dspy.InputField(desc="o texto base")
    texto: str = dspy.OutputField(desc="o exemplo sintético da sentença")
    # reasoning: str = dspy.OutputField(
    #     desc="A lógica que o levou a gerar a nova sentença"
    # )


class Gerador(dspy.Module):
    def __init__(self, signature=GeradorSignature):
        super().__init__()
        self.cot = dspy.ChainOfThought(
            signature,
        )

    def forward(self, TEXTO):
        return self.cot(texto_base=TEXTO)


def metrica_acertos(exemplo, pred, trace=None):
    print(pred)
    score = 0
    if (
        Tipo.embargo.value in exemplo.TIPO
        and Tipo.embargo.value == pred.classificao.tipo
    ):
        score += 2 if exemplo.SENTENCA == pred.classificacao.sentenca else -0.5
    else:
        score += int(exemplo.TIPO == pred.classificao.tipo) + int(
            exemplo.SENTECA == pred.classificao.sentenca
        )
    return score


def main():
    dl = DataLoader()
    df = pd.read_csv("./decisoes.csv")

    df = df.groupby(["TIPO", "SENTENCA"], group_keys=False).apply(
        lambda x: x.sample(min(len(x), 10))
    )
    df = dl.from_pandas(df, input_keys=("TEXTO",))
    print(df[0].inputs())
    dspy.configure(lm=llm)

    gerador = Gerador()
    # teleprompter = BootstrapFewShot(metric=metrica_acertos)
    # compiled = teleprompter.compile(gerador, trainset=df)
    # compiled.save("gerador.json")

    print(gerador(TEXTO=df[0].TEXTO))


if __name__ == "__main__":
    main()
