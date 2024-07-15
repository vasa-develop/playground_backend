from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import uuid
from Bio import SeqIO
import together
import subprocess
import torch
from dotenv import load_dotenv
from transformers import (
  AutoTokenizer,
  EsmForProteinFolding,
  set_seed
)
from typing import List

load_dotenv()  # Load environment variables from .env file

app = FastAPI()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

class ModelInput(BaseModel):
    prompt: str = (
        "|" # -> start prompt
        +"d__Bacteria;"
        +"p__Pseudomonadota;"
        +"c__Gammaproteobacteria;"
        +"o__Enterobacterales;"
        +"f__Enterobacteriaceae;"
        +"g__Escherichia;"
        +"s__Escherichia"
        +"|" # -> end prompt
    )
    max_tokens: int = 1024
    temperature: float = 1.0
    top_k: int = 4
    top_p: float = 1.0
    logprobs: bool = False

class ProteinInput(BaseModel):
    dna_sequence: str = "GCCGAGCCCCAGCCCAGGTACACGAAGCACCCGGAGCGCGCGAGCGCCCGATCTAGCATCCGCCCCCCCCCCCCCCCTCTCATCTGCCGCGGGCTGCGCACGACCGGCATCCGGCGCCCGCGTAGCACCCCCCCCCCCCCAATGCCACGCGCTGGGCGCTCATGGCTATCTCCATCCCACGCCCATTCGTTTGTTCGCCAAAGGGGxCGCGCGTAGCCCGCCCGGGGGCCGCCCGCCTTCGTCGCCGCCACTCGCCCGGGGCGGCCCGCCGCTCGGGGCGGCCGGGCGCCGGCTGTTCCCTCAGCGCCCTCCGCCGCTCCCAGGGGGGGTCCCGCGTGCGCCGCCCTGCTTTTCCGGCGCGCGGCGCGGCCGCAGTCAACGCGCCGCTCCTCGCACATGGTCATCCACCCGCGCGCATCTGAGCCGCCACAGCGGCCCAAGGCCACCCTCGTCTCGCGCTTGCCTCCCGGGGAGTCCCGGCTCTGCCTCCGTCATCTCCCCTGCCCCCCCGCACCGCCCCCCCCCCCCAGGCACCCCCGGGCCCCAACATACGCCGTCCGGCTGTACCTTTCTGCCTCGCTTATGCGGTAGCCTTTGTATCCGTCGACACTAGGGCGCCGGCCTGTCGCACGGTGCACCAGATGCTCCCGCACGGTTTTCGCCGCCTTTTCTTCCACCCGTACCCCCCCCCCCCCCTAGCCCGCCCCGCCTCCCGCTTGCCCGCCCGCCCCGTCGCCCGCAGTTGGGGTGCCCTCTTACCGGGGCGTGGCTGCGCGCCCGATTTGCCCCTGGCTCCCCACGCGCGGTCACTATCCCCCTGCGTTCAACGCGGGGCTCCCCCCCCATACTCCTTCCACCCCCTCCGGCAAGGAGCGTCGCGCGCGCCCGGCAGCTCTTTACCCGCATCTTTCCGCGCGCGGCAGCAGCCGCCTCCACCCAGGGGGCGCCCTTTCCTCCCGCGTTCTGCATTGCTGGTCCAGGGCGGCCCTTTCCGCCAGTTGGGGCGCCCGTTTCTTTTCTGCCC"

class ProteinOutput(BaseModel):
    protein_sequences: str

def delete_files(files: List[str]):
    for file in files:
        os.remove(file)

@app.post("/generate_dna_protein_structure")
async def generate_dna_protein_structure(input: ModelInput):
    together.api_key = TOGETHER_API_KEY
    output = together.Complete.create(
        prompt = input.prompt,
        model = 'togethercomputer/evo-1-131k-base',
        max_tokens = input.max_tokens,
        temperature = input.temperature,
        top_k = input.top_k,
        top_p = input.top_p,
        logprobs = input.logprobs
    )
    gen_dna_seq = output['choices'][0]['text']


    unique_id = uuid.uuid4().hex
    temp_fasta_path = f"temp_{unique_id}.faa"
    proteins_faa_path = f"proteins_{unique_id}.faa"
    genes_gff_path = f"genes_{unique_id}.gff"

    with open(temp_fasta_path, "w") as temp_fasta:
        temp_fasta.write(f">sequence\n{gen_dna_seq}")
    
    result = subprocess.run(
        ["prodigal", "-i", temp_fasta_path, "-a", proteins_faa_path, "-o", genes_gff_path, "-p", "meta", "-f", "gff"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        delete_files([temp_fasta_path, proteins_faa_path, genes_gff_path])
        raise HTTPException(status_code=400, detail="Prodigal failed")
    
    with open(proteins_faa_path) as f:
        protein_sequences = f.read()
    
    delete_files([temp_fasta_path, proteins_faa_path, genes_gff_path])




    esmfold = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1",
        low_cpu_mem_usage = True # we set this flag to save some RAM during loading
    )
    esmfold.esm = esmfold.esm.half() # -> make sure we use a lightweight precision

    # We will also need the ESMFold tokenizer to map our predicted protein-coding
    # genes to the token vocabulary of ESMFold:
    esmfold_tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

    # Using ESMFold, we can easily predict a protein folding structure:
    
    # write proteins_fasta file with a unique name to avoid race conditions
    unique_id = uuid.uuid4().hex
    proteins_faa_path = f"proteins_{unique_id}.faa"
    with open(proteins_faa_path, "w") as f:
        f.write(protein_sequences)

    # 1. Select a predicted protein-coding gene that we want to fold:
    i = 0 # lets just use the first one.
    protein_record = list(SeqIO.parse(proteins_faa_path, "fasta"))[i]
    protein_seq = str(protein_record.seq)[:-1] # remove stop codon

    # 2. Tokenize the gene to make it digestible for ESMFold:
    esmfold_in = esmfold_tokenizer(
        [protein_seq],
        return_tensors="pt",
        add_special_tokens=False
    )

    # 3. Feed the tokenized sequence to ESMfold:
    # (in PyTorch's inference_mode to avoid any costly gradient computations)
    with torch.inference_mode():
        esmfold_out = esmfold(**esmfold_in)
        esmfold_out_pdb = esmfold.output_to_pdb(esmfold_out)[0]

    return {
        "dna_sequence": gen_dna_seq,
        "protein_sequences": protein_sequences,
        "pdb_str": esmfold_out_pdb
    }

@app.post("/generate_dna")
async def generate_dna(input: ModelInput):
    together.api_key = TOGETHER_API_KEY
    output = together.Complete.create(
        prompt = input.prompt,
        model = 'togethercomputer/evo-1-131k-base',
        max_tokens = input.max_tokens,
        temperature = input.temperature,
        top_k = input.top_k,
        top_p = input.top_p,
        logprobs = input.logprobs
    )
    gen_dna_seq = output['choices'][0]['text']
    return {"dna_sequence": gen_dna_seq}

@app.post("/predict_genes")
async def predict_genes(input: ProteinInput):
    unique_id = uuid.uuid4().hex
    temp_fasta_path = f"temp_{unique_id}.faa"
    proteins_faa_path = f"proteins_{unique_id}.faa"
    genes_gff_path = f"genes_{unique_id}.gff"

    with open(temp_fasta_path, "w") as temp_fasta:
        temp_fasta.write(f">sequence\n{input.dna_sequence}")
    
    result = subprocess.run(
        ["prodigal", "-i", temp_fasta_path, "-a", proteins_faa_path, "-o", genes_gff_path, "-p", "meta", "-f", "gff"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        delete_files([temp_fasta_path, proteins_faa_path, genes_gff_path])
        raise HTTPException(status_code=400, detail="Prodigal failed")
    
    with open(proteins_faa_path) as f:
        protein_sequences = f.read()
    
    delete_files([temp_fasta_path, proteins_faa_path, genes_gff_path])
    return {"protein_sequences": protein_sequences}

@app.post("/predict_protein_structure")
async def predict_protein_structure(input: ProteinOutput):    
    esmfold = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1",
        low_cpu_mem_usage = True # we set this flag to save some RAM during loading
    )
    esmfold.esm = esmfold.esm.half() # -> make sure we use a lightweight precision

    # We will also need the ESMFold tokenizer to map our predicted protein-coding
    # genes to the token vocabulary of ESMFold:
    esmfold_tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

    # Using ESMFold, we can easily predict a protein folding structure:

    # write proteins_fasta file with a unique name to avoid race conditions
    unique_id = uuid.uuid4().hex
    proteins_faa_path = f"proteins_{unique_id}.faa"
    with open(proteins_faa_path, "w") as f:
        f.write(input.protein_sequences)

    # 1. Select a predicted protein-coding gene that we want to fold:
    i = 0 # lets just use the first one.
    protein_record = list(SeqIO.parse(proteins_faa_path, "fasta"))[i]
    protein_seq = str(protein_record.seq)[:-1] # remove stop codon

    # 2. Tokenize the gene to make it digestible for ESMFold:
    esmfold_in = esmfold_tokenizer(
        [protein_seq],
        return_tensors="pt",
        add_special_tokens=False
    )

    # 3. Feed the tokenized sequence to ESMfold:
    # (in PyTorch's inference_mode to avoid any costly gradient computations)
    with torch.inference_mode():
        esmfold_out = esmfold(**esmfold_in)
        esmfold_out_pdb = esmfold.output_to_pdb(esmfold_out)[0]

    return {"pdb_file": esmfold_out_pdb}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)