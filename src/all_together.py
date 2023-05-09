# pylint: disable=missing-function-docstring, missing-module-docstring, invalid-name
import asyncio
import pickle
from json import JSONDecodeError
from pathlib import Path

import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm_asyncio


def clean_names(
    mapped_csv: str | Path = Path("data", "Prescriptions_60pct_mapped.csv"),
    cleaned_csv: None | str | Path = Path("data", "rxnorm", "cleaned_names.csv"),
) -> pd.DataFrame:
    print(f"Reading {mapped_csv}")
    df_raw = pd.read_csv(
        mapped_csv,
        delimiter=";",
        usecols=["drug_name", "ATC_codes"],
    )

    # Filter to missing ATC Codes
    df_raw = df_raw.loc[df_raw.ATC_codes.isna(), :]

    # Clean drug names
    df = df_raw.copy()

    # To title case
    df.loc[:, "clean_name"] = df.loc[:, "drug_name"].str.title()

    # Remove special characters
    df.loc[:, "clean_name"] = df.loc[:, "clean_name"].str.replace(
        r"[\(\)/*\[\]\,]", " ", regex=True
    )

    # Remove multiple spaces and trailing spaces
    df.loc[:, "clean_name"] = df.loc[:, "clean_name"].str.replace(
        r"\s+", " ", regex=True
    )
    df.loc[:, "clean_name"] = df.loc[:, "clean_name"].str.strip()
    df.drop(["ATC_codes"], axis=1, inplace=True)

    print(f"Number of missing ATC codes: {len(df)}")

    if cleaned_csv is not None:
        print(f"Writing {cleaned_csv}")
        df.to_csv(cleaned_csv, index=False)

    return df


def get_rxnorm_codes(
    clean_df: pd.DataFrame,
    pickle_path: None | str | Path = Path("pickles", "rxdict.pkl"),
) -> dict:
    unique_names = clean_df.clean_name.unique()
    print(f"Number of unique drug names: {len(unique_names)}")

    async def fetch(session, code):
        params = {"term": code, "option": "1"}
        url = "https://rxnav.nlm.nih.gov/REST/approximateTerm.json"
        async with session.get(url, params=params) as response:
            try:
                return await response.json()
            except JSONDecodeError:
                return {}

    async def m(codes):
        session_timeout = aiohttp.ClientTimeout(total=None)
        no_timeout = aiohttp.ClientSession(timeout=session_timeout)
        async with no_timeout as session:
            tasks = []
            for code in codes:
                task = asyncio.ensure_future(fetch(session, code))
                tasks.append(task)
            responses = await tqdm_asyncio.gather(*tasks)
            return responses

    if not Path(pickle_path).exists():
        print("Running main for get_rxnorm_codes")
        rxnorms = asyncio.run(m(unique_names))
        rxnormdict = dict(zip(unique_names, rxnorms))

        if pickle_path is not None:
            print(f"Writing {pickle_path}")
            with open(pickle_path, "wb") as f:
                pickle.dump(rxnormdict, f)
    else:
        print(f"Found {pickle_path}")
        rxnormdict = pickle.load(open(pickle_path, "rb"))

    return rxnormdict


def wrangle_rxnorm_codes_to_df(
    rxnorms: dict, rxcui_path: None | str | Path = Path("data", "rxnorm", "rxcuis.csv")
) -> pd.DataFrame:
    with_codes = {}
    for k, val in rxnorms.items():
        try:
            with_codes[k] = val["approximateGroup"]["candidate"][0]["rxcui"]
        except (KeyError, IndexError):
            continue

    print(f"Number of rxnorm codes: {len(with_codes)}")

    df = pd.DataFrame.from_dict(with_codes, orient="index", columns=["rxcui"])

    if rxcui_path is not None:
        print(f"Writing {rxcui_path}")
        df.to_csv(rxcui_path, index_label="clean_name")

    return df


def get_rxcui(
    rxcui_df: pd.DataFrame,
    pickle_path: None | str | Path = Path("pickles", "rxdict_related.pkl"),
) -> dict:
    unique_rxcui = rxcui_df.rxcui.unique()
    print(f"Number of unique rxcui: {len(unique_rxcui)}")

    async def fetch(session, code):
        url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{code}/allrelated.json"
        async with session.get(url) as response:
            try:
                return await response.json()
            except JSONDecodeError:
                return {}

    async def m(codes):
        session_timeout = aiohttp.ClientTimeout(total=None)
        no_timeout = aiohttp.ClientSession(timeout=session_timeout)
        async with no_timeout as session:
            tasks = []
            for code in codes:
                task = asyncio.ensure_future(fetch(session, code))
                tasks.append(task)
            responses = await tqdm_asyncio.gather(*tasks)
            return responses

    if not Path(pickle_path).exists():
        # Run main
        print("Running main for get_rxcui")
        related = asyncio.run(m(unique_rxcui))
        rxdict = dict(zip(unique_rxcui, related))

        with open(pickle_path, "wb") as f:
            print(f"Writing {pickle_path}")
            pickle.dump(rxdict, f)
    else:
        print(f"Found {pickle_path}")
        rxdict = pickle.load(open(pickle_path, "rb"))

    return rxdict


def wrangle_rxcui_to_df(
    rxdict: dict,
    rxcui_path: None | str | Path = Path("data", "rxnorm", "rxcui_related.csv"),
) -> pd.DataFrame:
    new_d = {i: {} for i in rxdict.keys()}
    for i, j in rxdict.items():
        if "allRelatedGroup" in j:
            for k in j["allRelatedGroup"]["conceptGroup"]:
                if "conceptProperties" in k:
                    new_d[i][k["tty"]] = {
                        "rxcui": k["conceptProperties"][0]["rxcui"],
                        "name": k["conceptProperties"][0]["name"],
                    }
                else:
                    new_d[i][k["tty"]] = {"rxcui": "", "name": ""}
    # %%

    df = pd.DataFrame.from_dict(new_d, orient="index")
    # %%
    df = pd.melt(df, ignore_index=False)
    df = pd.concat([df.drop(["value"], axis=1), df["value"].apply(pd.Series)], axis=1)
    # %%
    df = df.loc[df.variable.isin(["IN", "PIN", "MIN"]) & df.rxcui.ne(""), :]

    print(f"Number of rxcui: {len(df)}")

    if rxcui_path is not None:
        print(f"Writing {rxcui_path}")
        df.to_csv(
            rxcui_path,
            index=True,
            header=True,
            index_label="original_rxcui",
        )

    return df


def get_atc(
    rxcui_df: pd.DataFrame, pickle_path: None | str | Path = Path("pickles", "atc.pkl")
) -> dict:
    unique_rxcui = rxcui_df.rxcui.unique()

    print(f"Number of unique rxcui: {len(unique_rxcui)}")

    # Define get_atc functions
    async def fetch(session, code):
        url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{code}/property.json?propName=ATC"
        async with session.get(url) as response:
            try:
                return await response.json()
            except JSONDecodeError:
                return {}

    async def m(codes):
        session_timeout = aiohttp.ClientTimeout(total=None)
        no_timeout = aiohttp.ClientSession(timeout=session_timeout)
        async with no_timeout as session:
            tasks = []
            for code in codes:
                task = asyncio.ensure_future(fetch(session, code))
                tasks.append(task)
            responses = await tqdm_asyncio.gather(*tasks)
            return responses

    if not Path(pickle_path).exists():
        # Run main
        print("Running main for get_atc")
        atc_codes = asyncio.run(m(unique_rxcui))
        rxdict = dict(zip(unique_rxcui, atc_codes))

        if pickle_path is not None:
            print(f"Writing {pickle_path}")
            with open(pickle_path, "wb") as f:
                pickle.dump(rxdict, f)
    else:
        print(f"Found {pickle_path}")
        rxdict = pickle.load(open(pickle_path, "rb"))

    return rxdict


def wrangle_atc_df(
    rxdict: dict, csv_path: None | str | Path = Path("data", "rxnorm", "rxcui_atc.csv")
) -> pd.DataFrame:
    test_d = {}
    for i, j in rxdict.items():
        prop_list = []
        if "propConceptGroup" in j:
            for k in j["propConceptGroup"]["propConcept"]:
                if "propValue" in k:
                    prop_list.append(k["propValue"])
                else:
                    prop_list.append("")
        test_d[i] = prop_list
    # %%
    atc_df = pd.DataFrame(test_d.items(), columns=["rxcui", "atc"]).set_index("rxcui")
    # %%
    atc_df = atc_df.loc[atc_df.atc.apply(lambda x: len(x) > 0), :]

    print(f"Number of rxcui with atc: {len(atc_df)}")

    if csv_path is not None:
        print(f"Writing {csv_path}")
        atc_df.to_csv(csv_path, index=True, header=True, index_label="rxcui")

    return atc_df


def final_join(
    clean_df: pd.DataFrame,
    rxcui_df: pd.DataFrame,
    atc_df: pd.DataFrame,
    csv_path: str | Path = Path("data", "final.csv"),
):
    clean_df = clean_df.merge(
        rxcui_df, how="left", left_on="clean_name", right_index=True
    )
    clean_df = clean_df.merge(atc_df, how="left", left_on="rxcui_y", right_index=True)

    print(f"Number of unique drugs: {len(clean_df)}")

    if csv_path is not None:
        print(f"Writing {csv_path}")
        clean_df.to_csv(csv_path, index=False)

    return clean_df


def main():
    clean_df = clean_names()
    rxnorm_codes = get_rxnorm_codes(clean_df)
    rxnorm_df = wrangle_rxnorm_codes_to_df(rxnorm_codes)
    rxdict = get_rxcui(rxnorm_df)
    rxcui_df = wrangle_rxcui_to_df(rxdict)
    atc_codes = get_atc(rxcui_df)
    atc_df = wrangle_atc_df(atc_codes)
    final_df = final_join(clean_df, rxcui_df, atc_df)

    return final_df


if __name__ == "__main__":
    result = main()
