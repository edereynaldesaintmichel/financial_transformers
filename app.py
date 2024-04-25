import json
import numpy as np


accepted_keys = ["Q2 2019",  "Q3 2019",  "Q4 2019",  "Q1 2020",  "Q2 2020",  "Q3 2020",  "Q4 2020",  "Q1 2021",  "Q2 2021",  "Q3 2021",  "Q4 2021",  "Q1 2022",  "Q2 2022",  "Q3 2022",  "Q4 2022",  "Q1 2023",  "Q2 2023"]


with open('formatted_data.json', 'r') as file:
    data = json.load(file)

# with open('only_one_company.json', 'r') as file:
#     company_data = json.load(file)

with open("leaves_2.json", "r") as file2:
    definition_vector = json.load(file2)


def get_all_leaves(one_company_data, prefix=""):
    leaves = []
    for key, value in one_company_data.items():
        if isinstance(value, str):
            leaves.append(f"{prefix}.{key}"[1:])
            continue
        leaves += get_all_leaves(value, prefix=f"{prefix}.{key}")

    return leaves

def get_all_aggregates(one_company_data, prefix=""):
    aggregates = []
    for key, value in one_company_data.items():
        if "value" in value:
            aggregates.append(f"{prefix}.{key}"[1:])
            continue
        if isinstance(value, str):
            continue
        aggregates += get_all_aggregates(value, prefix=f"{prefix}.{key}")

    return aggregates


def date_to_float(quarter_date: str):
    sub_dates = quarter_date.split(sep=" ")
    date = float(sub_dates[1])
    date += float((float(sub_dates[0][1]) - 1) * 0.25)

    return date


def sanitize_value(value):
    if isinstance(value, float) or isinstance(value, int):
        return value
    return 0


def get_tmp_company_data(company_data, definition_vector):
    vector = []
    for key in definition_vector:
        tmp = company_data
        for sub_key in key.split(sep="."):
            tmp = tmp.get(sub_key, {})
        vector.append([sanitize_value(tmp.get('value', {}).get(key, None))
                       for key in accepted_keys])
    vector.append([date_to_float(key)
                   for key in accepted_keys])

    return vector


def get_vectors(data, definition_vector):
    vectors = []
    for company_data in data:
        # try:
            vector = get_tmp_company_data(company_data, definition_vector)
            vectors.append(vector)
        # except Exception as e:
        #     print("key was not present, skipping company")

    return np.array(vectors).transpose((0, 2, 1)).tolist()


vectors = get_vectors(data, definition_vector)

with open("leaves_training_data.json", "w+") as file:
    file.write(json.dumps(vectors, indent=4))

"""
#TODO: Importer le transformer d'Andrej Karpathy. 
Tester tel quel, (juste adapté aux chiffres et à la régression plutôt qu'à la classification)
#TODO: Ensuite, je crains que les 0 intempestifs dans le training data niquent le travail des têtes d'attention, qui sont avant tout linéaires. 
Il faut que je voie comment gérer ces 0 de façon pas du tout du tout linéaire dans un méchanisme d'attention custom.
"""
