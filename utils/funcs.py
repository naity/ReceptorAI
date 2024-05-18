import os
import re
import torch
import requests
import py3Dmol
import subprocess
import pandas as pd
import numpy as np
import streamlit as st

from utils import *
from tqdm import tqdm
from pinecone import Pinecone
from transformers import AutoTokenizer, EsmModel


def extract_substring_between_parentheses(s):
    """Extracts the substring between parentheses from a string using regular expressions.

    Args:
      string: The string to extract the substring from.

    Returns:
      The substring between parentheses, or the original string if there are no
      parentheses in the string.
    """

    pattern = r"\(([^)]+)\)"
    match = re.search(pattern, s)
    if match:
        return match.group(1)
    else:
        return s


@st.cache_resource(show_spinner="Loading models...")
def get_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    return tokenizer, model


def get_embeddings(tokenizer, model, seqs):
    """Get embeddings for a list of sequences."""
    inputs = tokenizer(seqs, padding="longest", return_tensors="pt")
    batch_lens = (inputs["input_ids"] != tokenizer.get_vocab()["<pad>"]).sum(dim=1)

    with torch.no_grad():
        results = model(**inputs)
    token_representations = results.last_hidden_state

    # Generate per-sequence representations via averaging
    # First token is CLS, last token is EOS
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(
            token_representations[i, 1 : tokens_len - 1].mean(0).numpy(force=True)
        )

    return np.array(sequence_representations)


def upsert_to_index(pc_index, records, batch_size=100):
    """
    Upsert items to a Pinecone index.

    Args:
        pc_index (PineconeIndex): The Pinecone index to upsert items to.
        records (list): A list of dictionaries containing the items to upsert.
        batch_size (int): The batch size for upserting items.
    """
    for i in tqdm(range(0, len(records), batch_size)):
        pc_index.upsert(vectors=records[i : i + batch_size])


def search_embedding(embedding, top_k, species, pc_index=None):
    """Search a single embedding against the TCR vector database.

    Args:
        embedding (np.array): The embedding to search.
        top_k (int): The number of top results to return.
        species (str): The species to filter by.
        pc_index (PineconeIndex): The Pinecone index to search.
    Returns:
        dict: A dictionary containing the search results.
    """
    if pc_index is None:
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        pc_index = pc.Index("tcrs")

    results = pc_index.query(
        vector=embedding,
        top_k=top_k,
        include_values=False,
        filter={"Species": {"$eq": species}},
        include_metadata=True,
    )

    # return matches in lists
    ids = []
    metadatas = []
    scores = []

    for match in results["matches"]:
        ids.append(match["id"])
        metadatas.append(match["metadata"])
        scores.append(match["score"])

    return {"ids": ids, "metadatas": metadatas, "scores": scores}


def format_result(tcr_matches):
    """format response"""
    df = pd.DataFrame(tcr_matches["metadatas"])
    df["Similarity Score"] = [min(1, 1 - d) for d in tcr_matches["scores"]]

    # rearrange columns
    df = df[
        [
            "Similarity Score",
            "Species",
            "Antigen Epitope",
            "Antigen Protein",
            "Antigen Source",
            "CDR3.beta.aa",
            "TRBV",
            "TRBJ",
            "Reference",
            "Database",
        ]
    ]
    return df.sort_values("Similarity Score", ascending=False)


def run_stitchr(tcr, verbose=False):
    species = tcr["Species"].lower()

    v = tcr["TRBV"]
    j = tcr["TRBJ"]
    cdr3 = tcr["CDR3.beta.aa"]

    if species not in ("human", "mouse"):
        if verbose:
            print("Species not supported.")
        return None

    # empyt strings
    if v == "" or j == "" or cdr3 == "":
        if verbose:
            print("Missing values.")
        return None

    cmd = f"stitchr -s {species} -v {v} -j {j} -cdr3 {cdr3} -m AA"
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True, check=True)
    except Exception as e:
        if verbose:
            print(e)
        # return None if stitchr fails
        return None
    return result.stdout.decode("utf-8").strip()


def fold_sequence(sequence):
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }

    return requests.post(
        "https://api.esmatlas.com/foldSequence/v1/pdb/",
        headers=headers,
        data=sequence,
        verify=False,
    )


def generate_3d_view(response):
    view = py3Dmol.view()
    view.addModelsAsFrames(response.text)
    view.setStyle({"model": -1}, {"cartoon": {"color": "spectrum"}})
    view.zoomTo()
    return view
