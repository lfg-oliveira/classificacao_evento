#!/usr/bin/env python3

from dsp import Optional
import dspy
from dspy.teleprompt import MIPRO, BootstrapFewShotWithRandomSearch, MIPROv2
import pandas as pd
from dspy.datasets import DataLoader
from enum import Enum
import pydantic
from pydantic_core.core_schema import ExpectedSerializationTypes
import pickle

# Phoenix setup
# Se estiver em um Jupyter Notebook
# ==
# import phoenix as px


# phoenix_session = px.
def setup_phoenix():
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


# DSPy Stuff

NUM_THREADS = 4

# df = pd.read_csv("./bs2.csv").fillna(0).astype(str).sample(frac=0.33)
# example = df.head(1).movimentacao.values[0]
ollama_model = dspy.OllamaLocal(
    model="mistral-nemo",
)

teacher = ollama_model
# dataset = [
#     dspy.Example(**data).with_inputs("movimentacao")
#     for (_, data) in df.drop(columns=["OBS"]).to_dict(orient="index").items()
# ]

dataset = DataLoader().from_csv(
    "./20240724_bs2.csv",
    input_keys=("movimentacao",),
)
dataset = DataLoader().sample(dataset, 10)
dspy.settings.configure(lm=ollama_model)

with open("sict.pkl", "wb") as fd:
    pickle.dump(dataset, fd)

# example_dataset = [dspy.Example(**example) for example in df.to_dict()]


class Booleano(Enum):
    yes = "SIM"
    no = "NÃO"


class Resposta(pydantic.BaseModel):
    dano_moral: Optional[bool]
    valor_dano_moral: Optional[str]
    dano_material: Optional[bool]
    valor_dano_material: Optional[str]


class RespostaGuardrail(dspy.Signature):
    """Por favor, verfique se a entrada segue o padrão de valores separados por vírgula (CSV)."""

    entrada = dspy.InputField()
    formatado_corretamente: Booleano = dspy.OutputField(
        desc="retorna somente SIM ou NÃO para verdadeiro e falso, respectivamente"
    )
    motivo_decisao_formatado_corretamente = dspy.OutputField()


class RetryResposta(dspy.Signature):
    """Dados a entrada incial e o motivo da falha na checagem, corrija-o.
    Retire qualquer texto que apareceu antes dos valores requisitados como 'Claro, aqui estão os dados em formato CSV:'.
    Os campos de 'valor_alguma_coisa' devem ser número. Trocar 'null' ou 'Não mencionado' pelo número 0.
    """

    entrada_inicial = dspy.InputField()
    motivo_erro = dspy.InputField()
    resposta = dspy.OutputField()


class MovimentacaoClassificada(dspy.Signature):
    """Sua tarefa é extrair os dados da movimentação abaixo seguindo o esquema dos campos. Me fale há presença de dano moral ou material e seus respectivos valores. Para identificar valores, utilize contextos como 'no valor de R$ {valor numérico}'"""

    movimentacao: str = dspy.InputField()
    resposta: Resposta = dspy.OutputField()


# class FactJudge(dspy.Signature):
#     verdade = dspy.InputField()
#     predicao = dspy.InputField()
#     predicao_correta = dspy.OutputField(
#         desc="A predição está semelhante a resposta, mesmo que não necessáriamente idêntica?",
#         prefix="Correta[Sim/Não]",
#     )


class Classificador(dspy.Module):
    def __init__(self, signature):
        self.saida_csv = dspy.functional.TypedPredictor(
            signature, max_retries=6, explain_errors=True
        )

    def forward(self, movimentacao: str):
        data = self.saida_csv(movimentacao=movimentacao)
        return data


def metrica_extracao(exemplo: dspy.Example, pred: dspy.Prediction, trace=None):
    score = 0
    for key in exemplo.labels():
        score += int((pred.get(key) or 0) == (exemplo.get(key) or 0))
    return score


if __name__ == "__main__":
    setup_phoenix()
    modelo = Classificador(MovimentacaoClassificada)

    splits = DataLoader().train_test_split(
        dataset=dataset,
        train_size=0.8,
    )
    train_dataset = splits["train"]
    test_dataset = splits["test"]

    optimizer = MIPRO(
        metric=metrica_extracao,
        task_model=ollama_model,
        prompt_model=teacher,
        # teacher_settings={"lm": teacher},
    )

    eval_kwargs = dict(num_threads=NUM_THREADS, display_progress=True, display_table=0)
    try:
        compilado = optimizer.compile(
            modelo,
            trainset=train_dataset,
            max_bootstrapped_demos=4,
            max_labeled_demos=4,
            num_trials=5,
            eval_kwargs=eval_kwargs,
        )
        compilado.save("classificador.json")
    except IndexError as e:
        print(e)
# Para ver o prompt que foi enviado
# print(ollama_model.inspect_history(n=1))
# ------------------------------------------------------
# modelo.load("classificador.json")
# pred = modelo(
#     "Nº 1027121-20.2023.8.26.0506 - Processo Digital. Petições para juntada devem ser apresentadas exclusivamente por meio eletrônico, nos termos do artigo 7º da Res. 551/2011 - Apelação Cível - Ribeirão Preto - Apte/Apda: Claudiane Santos Alves Ferreira - Apdo/Apte: Mrv Engenharia e Participações S.a. - Magistrado(a) João Pazine Neto - Deram parcial provimento ao recurso da Autora e negaram provimento ao recurso da Ré. V.U.  - INDENIZAÇÃO POR DANOS MORAIS E MATERIAIS. VENDA E COMPRA DE IMÓVEL. BEM ENTREGUE SEM A INFRAESTRUTURA DE ABASTECIMENTO DE ÁGUA E SANEAMENTO BÁSICO. FORNECIMENTO PELA RÉ DE ÁGUA IMPRÓPRIA PARA O CONSUMO, POR MEIO CAMINHÕES PIPA. DANO MORAL CARACTERIZADO E MAJORADO A R$ 15.000,00, COM ATUALIZAÇÃO MONETÁRIA DESSA FIXAÇÃO E JUROS A CONTAR DA CITAÇÃO, AFASTADA A APLICAÇÃO DA TAXA SELIC. VERBA HONORÁRIA DEVIDA PELA RÉ MAJORADA. RECURSO DA AUTORA PARCIALMENTE PROVIDO E NÃO PROVIDO O DA RÉ. ART. 1007 CPC - EVENTUAL RECURSO - SE AO STJ: CUSTAS R$ 247,14 - (GUIA GRU NO SITE http://www.stj.jus.br) - RESOLUÇÃO STJ/GP N. 2 DE 1º DE FEVEREIRO DE 2017; SE AO STF: CUSTAS R$ 223,79 - GUIA GRU - COBRANÇA - FICHA DE COMPENSAÇÃO - (EMITIDA ATRAVÉS DO SITE www.stf.jus.br) E PORTE DE REMESSA E RETORNO R$ 140,90 - GUIA FEDTJ - CÓD 140-6 - BANCO DO BRASIL OU INTERNET - RESOLUÇÃO N. 662 DE 10/02/2020 DO STF. Os valores referentes ao PORTE DE REMESSA E RETORNO, não se aplicam aos PROCESSOS ELETRÔNICOS, de acordo com o art. 4º, inciso II, da RESOLUÇÃO N. 662 DE 10/02/2020 DO STF. - Advs: Guilherme Mendonça Mendes de Oliveira (OAB: 331385/SP) - Fabiana Barbassa Luciano (OAB: 320144/SP) - Paula Stéphani Lorençato (OAB: 492340/SP) - Sala 803 - 8º ANDAR"
# )
# print(pred)
