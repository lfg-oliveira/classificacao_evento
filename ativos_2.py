#!/usr/bin/env python3

import pandas as pd

saida = pd.read_csv("saida.csv")
original = pd.read_csv("entrada_ativos.csv")
original = original.loc[original.numero_processo.isin(saida.num_process)]

original = original.rename(
    columns={
        "dano_moral": "dano_moral_original",
        "dano_material": "dano_material_original",
        "valor_dano_moral": "valor_dano_moral_original",
        "valor_dano_material": "valor_dano_material_original",
        "numero_processo": "num_process",
    }
)

final = saida.merge(
    original[
        [
            "num_process",
            "dano_moral_original",
            "dano_material_original",
            "valor_dano_moral_original",
            "valor_dano_material_original",
        ]
    ],
    how="left",
)

final.to_csv("saida_mergead.csv")
