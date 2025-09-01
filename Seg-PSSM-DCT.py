import os
import subprocess
import numpy as np
from Bio import SeqIO
from scipy.fftpack import dct

def run_psiblast(fasta_file, db, out_dir="pssm_out", num_iterations=3, evalue=0.001):
    """
    Runs PSI-BLAST to generate PSSM for each sequence in the FASTA file.
    Requires NCBI BLAST+ to be installed and a protein database (e.g., nr, swissprot).
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    pssm_files = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq_file = os.path.join(out_dir, f"{record.id}.fasta")
        with open(seq_file, "w") as f:
            f.write(f">{record.id}\n{str(record.seq)}\n")

        out_ascii = os.path.join(out_dir, f"{record.id}.pssm")
        cmd = [
            "psiblast",
            "-query", seq_file,
            "-db", db,
            "-num_iterations", str(num_iterations),
            "-evalue", str(evalue),
            "-out_ascii_pssm", out_ascii,
            "-out", os.path.join(out_dir, f"{record.id}.out")
        ]
        subprocess.run(cmd, check=True)
        pssm_files.append(out_ascii)
    return pssm_files

def read_pssm(pssm_file):
    """
    Reads PSSM ASCII file into numpy array.
    Returns: PSSM matrix of shape (L, 20), where L = sequence length.
    """
    matrix = []
    with open(pssm_file, "r") as f:
        start_reading = False
        for line in f:
            if line.strip().startswith("Last position-specific scoring matrix computed"):
                break
            if start_reading:
                parts = line.strip().split()
                if len(parts) < 22:
                    continue
                scores = list(map(int, parts[2:22]))
                matrix.append(scores)
            elif line.strip().startswith("1 "):
                start_reading = True
                parts = line.strip().split()
                scores = list(map(int, parts[2:22]))
                matrix.append(scores)
    return np.array(matrix)

def segment_pssm(pssm_matrix, segment_length=50):
    """
    Segments PSSM into non-overlapping windows.
    If sequence length not multiple of segment_length, last segment is padded with zeros.
    """
    L, D = pssm_matrix.shape
    num_segments = int(np.ceil(L / segment_length))
    segments = []
    for i in range(num_segments):
        start = i * segment_length
        end = min((i + 1) * segment_length, L)
        seg = pssm_matrix[start:end, :]
        if seg.shape[0] < segment_length:
            pad = np.zeros((segment_length - seg.shape[0], D))
            seg = np.vstack((seg, pad))
        segments.append(seg)
    return segments

def apply_dct(segments, k=20):
    """
    Applies DCT to each segment and keeps top-k coefficients for each column.
    Returns: Feature matrix (num_segments, 20*k).
    """
    features = []
    for seg in segments:
        seg_features = []
        for col in range(seg.shape[1]):
            coeffs = dct(seg[:, col], norm='ortho')
            seg_features.extend(coeffs[:k])
        features.append(seg_features)
    return np.array(features)

# Example workflow
if __name__ == "__main__":
    fasta_file = "example.fasta"
    db = "/path/to/blastdb/nr"  # replace with your BLAST DB path

    # Step 1: Run PSI-BLAST and get PSSMs
    # pssm_files = run_psiblast(fasta_file, db)

    # Step 2: For demonstration, assume one PSSM file
    # Example PSSM reading and processing
    # pssm = read_pssm(pssm_files[0])

    # DEMO with dummy PSSM (length=120, 20 columns)
    pssm = np.random.randint(-5, 10, size=(120, 20))

    # Step 3: Segment PSSM
    segments = segment_pssm(pssm, segment_length=50)

    # Step 4: Apply DCT
    features = apply_dct(segments, k=20)

    print("Feature shape:", features.shape)
    print(features)
