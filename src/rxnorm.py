# %%
import asyncio
import pickle
from json import JSONDecodeError
from pathlib import Path

import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

# pylint: disable=missing-function-docstring

# Read CSV and clean drug names
# %%
df_raw = pd.read_csv(
    Path("..", "data", "Prescriptions_60pct_mapped.csv"),
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
    r"[\(\)/*\[\]]", " ", regex=True
)

# Remove multiple spaces and trailing spaces
df.loc[:, "clean_name"] = df.loc[:, "clean_name"].str.replace(r"\s+", " ", regex=True)
df.loc[:, "clean_name"] = df.loc[:, "clean_name"].str.strip()
# %%

unique_names = df.clean_name.unique()
print(f"Number of unique drug names: {len(unique_names)}")
# %%


# Define rx_norm functions
async def fetch(session, code):
    params = {"term": code, "option": "1"}
    url = "https://rxnav.nlm.nih.gov/REST/approximateTerm.json"
    async with session.get(url, params=params) as response:
        try:
            return await response.json()
        except JSONDecodeError:
            return {}


async def main(codes):
    session_timeout = aiohttp.ClientTimeout(total=None)
    no_timeout = aiohttp.ClientSession(timeout=session_timeout)
    async with no_timeout as session:
        tasks = []
        for code in codes:
            task = asyncio.ensure_future(fetch(session, code))
            tasks.append(task)
        responses = await tqdm_asyncio.gather(*tasks)
        return responses


# Run main
rxnorms = asyncio.run(main(unique_names))
rxdict = dict(zip(unique_names, rxnorms))


# Save results
with open("pickles/rxdict.pkl", "wb") as f:
    pickle.dump(rxdict, f)
