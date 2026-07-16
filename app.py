"""
Nanopore Variant & CDR Analysis - Streamlit web app

Run locally with:
    pip install -r requirements.txt
    streamlit run app.py

Deploy publicly on Streamlit Community Cloud:
    1. Push this folder (app.py, requirements.txt, packages.txt) to a GitHub repo
    2. Go to https://share.streamlit.io -> New app -> point at your repo
    3. Streamlit Cloud reads packages.txt to apt-install minimap2 automatically

Note: this version deliberately avoids pysam. pysam is a compiled C-extension
(wraps htslib) and is only tested/supported up to Python 3.13 - Streamlit
Community Cloud has been intermittently forcing new deployments onto Python
3.14, which crashes pysam silently (segfault, no traceback). Since all we
actually need is a linear scan through minimap2's alignments (no BAM
sorting/random-access indexing required for this use case), we parse
minimap2's plain-text SAM output directly in pure Python instead. This also
means samtools is no longer a dependency at all.
"""

import os
import re
import subprocess
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import binom
import matplotlib
matplotlib.use("Agg")  # non-interactive backend - required for headless servers
import matplotlib.pyplot as plt

# =====================================================================
# CONSTANTS / SHARED LOGIC (same as the standalone scripts)
# =====================================================================
CODON_TABLE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L', 'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M', 'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S', 'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T', 'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*', 'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K', 'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W', 'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R', 'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}


def translate_codon(bases3):
    return CODON_TABLE.get(bases3.upper(), 'X')


def find_multi_bp_events(aligned_pairs, ref_seq, query_seq, min_len=2):
    """Group consecutive mismatch/deletion/insertion positions into single events."""
    events = []
    n = len(aligned_pairs)
    i = 0
    while i < n:
        query_pos, ref_pos = aligned_pairs[i]
        if ref_pos is not None and query_pos is not None:
            ref_base = ref_seq[ref_pos]
            query_base = query_seq[query_pos].upper()
            if query_base != ref_base:
                block_start = ref_pos
                block_end = ref_pos
                alt_bases = [query_base]
                j = i + 1
                while j < n:
                    qp2, rp2 = aligned_pairs[j]
                    if (rp2 is not None and qp2 is not None and rp2 == block_end + 1
                            and query_seq[qp2].upper() != ref_seq[rp2]):
                        alt_bases.append(query_seq[qp2].upper())
                        block_end = rp2
                        j += 1
                    else:
                        break
                if (block_end - block_start + 1) >= min_len:
                    events.append({"start": block_start, "end": block_end, "type": "SUB",
                                    "ref_block": ref_seq[block_start:block_end + 1],
                                    "alt_block": "".join(alt_bases)})
                i = j
                continue
            else:
                i += 1
                continue
        elif ref_pos is not None and query_pos is None:
            block_start = ref_pos
            block_end = ref_pos
            j = i + 1
            while j < n:
                qp2, rp2 = aligned_pairs[j]
                if rp2 is not None and qp2 is None and rp2 == block_end + 1:
                    block_end = rp2
                    j += 1
                else:
                    break
            if (block_end - block_start + 1) >= min_len:
                events.append({"start": block_start, "end": block_end, "type": "DEL",
                                "ref_block": ref_seq[block_start:block_end + 1],
                                "alt_block": "-" * (block_end - block_start + 1)})
            i = j
            continue
        elif ref_pos is None and query_pos is not None:
            anchor = aligned_pairs[i - 1][1] if i > 0 else None
            ins_bases = [query_seq[query_pos].upper()]
            j = i + 1
            while j < n:
                qp2, rp2 = aligned_pairs[j]
                if rp2 is None and qp2 is not None:
                    ins_bases.append(query_seq[qp2].upper())
                    j += 1
                else:
                    break
            if anchor is not None and len(ins_bases) >= min_len:
                events.append({"start": anchor, "end": anchor, "type": "INS",
                                "ref_block": "-", "alt_block": "".join(ins_bases)})
            i = j
            continue
        else:
            i += 1
    return events


def get_cdr_length(aligned_pairs, cdr_start, cdr_end):
    length = 0
    in_cdr = False
    for query_pos, ref_pos in aligned_pairs:
        if ref_pos is not None:
            if cdr_start <= ref_pos <= cdr_end:
                in_cdr = True
            elif ref_pos > cdr_end:
                in_cdr = False
        if in_cdr and query_pos is not None:
            length += 1
        if ref_pos is not None and ref_pos > cdr_end:
            break
    return length


def get_amino_acid_presence(aligned_pairs, ref_seq, query_seq, cdr_start0, cdr_end0):
    ref_to_query = {ref_pos: query_pos for query_pos, ref_pos in aligned_pairs if ref_pos is not None}
    presence = []
    n_codons = (cdr_end0 - cdr_start0 + 1) // 3
    for i in range(n_codons):
        p0, p1, p2 = cdr_start0 + 3 * i, cdr_start0 + 3 * i + 1, cdr_start0 + 3 * i + 2
        ref_codon = ref_seq[p0] + ref_seq[p1] + ref_seq[p2]
        ref_aa = translate_codon(ref_codon)
        qpos = [ref_to_query.get(p0), ref_to_query.get(p1), ref_to_query.get(p2)]
        if any(qp is None for qp in qpos):
            presence.append(0)
            continue
        read_codon = query_seq[qpos[0]] + query_seq[qpos[1]] + query_seq[qpos[2]]
        presence.append(1 if translate_codon(read_codon) == ref_aa else 0)
    return presence


# =====================================================================
# PURE-PYTHON REPLACEMENTS FOR PYSAM (no compiled C-extension dependency)
# =====================================================================
CIGAR_RE = re.compile(r'(\d+)([MIDNSHP=X])')


def sam_to_aligned_pairs(pos_1based, cigar_str):
    """
    Pure-Python equivalent of pysam's read.get_aligned_pairs(with_seq=False).
    Returns a list of (query_pos, ref_pos) tuples, 0-based, matching pysam's
    convention: soft/hard clips excluded entirely, insertions as
    (query_pos, None), deletions as (None, ref_pos), matches/mismatches as
    (query_pos, ref_pos).
    """
    query_pos = 0
    ref_pos = pos_1based - 1
    pairs = []
    for length_str, op in CIGAR_RE.findall(cigar_str):
        length = int(length_str)
        if op in ("M", "=", "X"):
            for _ in range(length):
                pairs.append((query_pos, ref_pos))
                query_pos += 1
                ref_pos += 1
        elif op == "I":
            for _ in range(length):
                pairs.append((query_pos, None))
                query_pos += 1
        elif op in ("D", "N"):
            for _ in range(length):
                pairs.append((None, ref_pos))
                ref_pos += 1
        elif op == "S":
            query_pos += length  # soft clip: consumes query, excluded from pairs
        # H and P consume neither query nor reference - nothing to do
    return pairs


def sam_reference_length(cigar_str):
    """Total reference span covered by this alignment (sum of M/D/N/=/X ops)."""
    total = 0
    for length_str, op in CIGAR_RE.findall(cigar_str):
        if op in ("M", "D", "N", "=", "X"):
            total += int(length_str)
    return total


def parse_sam_lines(sam_text):
    """
    Parse minimap2's SAM text output into a list of alignment dicts:
    {name, flag, pos, cigar, seq, is_unmapped, is_secondary, is_supplementary}
    Skips header lines (starting with '@').
    """
    records = []
    for line in sam_text.splitlines():
        if not line or line.startswith("@"):
            continue
        fields = line.split("\t")
        if len(fields) < 11:
            continue
        qname, flag_str, rname, pos_str, mapq, cigar = fields[0], fields[1], fields[2], fields[3], fields[4], fields[5]
        seq = fields[9]
        flag = int(flag_str)
        records.append({
            "name": qname,
            "flag": flag,
            "pos": int(pos_str),
            "cigar": cigar,
            "seq": seq,
            "is_unmapped": bool(flag & 4),
            "is_secondary": bool(flag & 256),
            "is_supplementary": bool(flag & 2048),
        })
    return records


# =====================================================================
# PIPELINE: alignment + filtering (runs once, cached in session_state)
# =====================================================================
def run_alignment_and_filter(reference_seq, fastq_bytes, min_coverage, min_homology):
    workdir = tempfile.mkdtemp()
    ref_fasta = os.path.join(workdir, "reference.fasta")
    fastq_path = os.path.join(workdir, "sample.fastq")

    ref_seq = reference_seq.strip().upper()
    with open(ref_fasta, "w") as f:
        f.write(">reference\n")
        f.write(ref_seq + "\n")

    with open(fastq_path, "wb") as f:
        f.write(fastq_bytes)

    # minimap2 writes SAM directly to stdout - no samtools sort/index needed,
    # since we only need a single linear pass through the alignments.
    align_cmd = ["minimap2", "-ax", "map-ont", ref_fasta, fastq_path]
    result = subprocess.run(align_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Alignment failed:\n{result.stderr}")

    sam_records = parse_sam_lines(result.stdout)
    ref_len = len(ref_seq)

    passed_reads = []  # list of dicts: {name, aligned_pairs, query_seq}
    total_processed = 0

    for rec in sam_records:
        if rec["is_unmapped"] or rec["is_secondary"] or rec["is_supplementary"]:
            continue
        total_processed += 1

        aligned_len = sam_reference_length(rec["cigar"])
        if aligned_len == 0:
            continue
        coverage = aligned_len / ref_len
        if coverage < min_coverage:
            continue

        aligned_pairs = sam_to_aligned_pairs(rec["pos"], rec["cigar"])
        query_seq = rec["seq"].upper()

        matches = 0
        compared = 0
        for query_pos, ref_pos in aligned_pairs:
            if ref_pos is None:
                continue
            compared += 1
            if query_pos is None:
                continue
            if query_seq[query_pos] == ref_seq[ref_pos]:
                matches += 1
        homology = (matches / compared) if compared else 0
        if homology < min_homology:
            continue

        passed_reads.append({"name": rec["name"], "aligned_pairs": aligned_pairs, "query_seq": query_seq,
                              "coverage": coverage, "homology": homology})

    return {
        "ref_seq": ref_seq,
        "ref_len": ref_len,
        "total_processed": total_processed,
        "passed_reads": passed_reads,
    }


# =====================================================================
# STREAMLIT UI
# =====================================================================
st.set_page_config(page_title="Nanopore Variant & CDR Analysis", page_icon="🎀", layout="wide")
st.title("🎀 Nanopore Variant & CDR Analysis")
st.caption("Upload a reference sequence and a FASTQ file, then run alignment, variant calling, "
           "and CDR-level distribution analyses. ✨")

with st.sidebar:
    st.header("1. Input")
    reference_seq = st.text_area("Reference sequence (nucleotides)", height=150,
                                  placeholder="Paste your reference sequence here...")
    fastq_file = st.file_uploader("FASTQ file", type=["fastq", "fq", "txt"])

    st.header("2. Quality filters")
    min_coverage = st.slider("Minimum coverage (fraction of reference length)", 0.50, 1.00, 0.92, 0.01)
    min_homology = st.slider("Minimum homology (fraction identity)", 0.50, 1.00, 0.82, 0.01)

    st.header("3. Variant reporting")
    variant_freq_threshold = st.slider("Variant frequency threshold", 0.01, 0.50, 0.05, 0.01)
    min_event_length = st.number_input("Minimum multi-bp event length", min_value=2, max_value=50, value=2)

    st.header("4. CDR regions (1-based nt, inclusive)")
    st.caption("Must be whole codons (length divisible by 3). Leave a region's start=0 to skip it.")
    cdr1_start = st.number_input("CDR1 start", min_value=0, value=0)
    cdr1_end = st.number_input("CDR1 end", min_value=0, value=0)
    cdr2_start = st.number_input("CDR2 start", min_value=0, value=0)
    cdr2_end = st.number_input("CDR2 end", min_value=0, value=0)
    cdr3_start = st.number_input("CDR3 start", min_value=0, value=0)
    cdr3_end = st.number_input("CDR3 end", min_value=0, value=0)

    run_button = st.button("Run alignment + filtering", type="primary")

cdr_regions = {}
for label, s, e in [("CDR1", cdr1_start, cdr1_end), ("CDR2", cdr2_start, cdr2_end), ("CDR3", cdr3_start, cdr3_end)]:
    if s > 0 and e > s and (e - s + 1) % 3 == 0:
        cdr_regions[label] = (s, e)

if run_button:
    if not reference_seq.strip():
        st.error("Please paste a reference sequence.")
    elif fastq_file is None:
        st.error("Please upload a FASTQ file.")
    else:
        with st.spinner("Running minimap2 alignment and filtering reads..."):
            try:
                result = run_alignment_and_filter(reference_seq, fastq_file.getvalue(), min_coverage, min_homology)
                st.session_state["pipeline_result"] = result
                st.session_state["cdr_regions"] = cdr_regions
                st.success(f"Done. {len(result['passed_reads'])} / {result['total_processed']} reads passed filters.")
            except Exception as e:
                st.error(str(e))

if "pipeline_result" in st.session_state:
    result = st.session_state["pipeline_result"]
    ref_seq = result["ref_seq"]
    ref_len = result["ref_len"]
    passed_reads = result["passed_reads"]
    cdr_regions = st.session_state.get("cdr_regions", {})

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Single-bp variants", "Multi-bp variants", "CDR length distribution", "CDR amino acid presence"]
    )

    # --- TAB 1: single-bp variant report ---
    with tab1:
        st.subheader("Single-bp variant report")
        variant_counts = {pos: {'A': 0, 'T': 0, 'C': 0, 'G': 0, 'Del': 0, 'Total': 0} for pos in range(ref_len)}
        for r in passed_reads:
            for query_pos, ref_pos in r["aligned_pairs"]:
                if ref_pos is None:
                    continue
                if query_pos is None:
                    variant_counts[ref_pos]['Del'] += 1
                    variant_counts[ref_pos]['Total'] += 1
                    continue
                qb = r["query_seq"][query_pos].upper()
                if qb in variant_counts[ref_pos]:
                    variant_counts[ref_pos][qb] += 1
                variant_counts[ref_pos]['Total'] += 1

        records = []
        for pos in range(ref_len):
            counts = variant_counts[pos]
            total_depth = counts['Total']
            if total_depth == 0:
                continue
            ref_base = ref_seq[pos]
            for base in ['A', 'T', 'C', 'G', 'Del']:
                count = counts[base]
                if base == ref_base or count == 0:
                    continue
                freq = count / total_depth
                if freq >= variant_freq_threshold:
                    records.append({"Position": pos + 1, "Reference_Base": ref_base, "Mutation_Base": base,
                                     "Read_Count": count, "Total_Depth": total_depth, "Frequency": round(freq, 4)})
        df_single = pd.DataFrame(records)
        if not df_single.empty:
            df_single = df_single.sort_values("Position")
            st.dataframe(df_single, use_container_width=True)
            st.download_button("Download CSV", df_single.to_csv(index=False), "variant_report.csv")
        else:
            st.info("No single-bp variants crossed the frequency threshold.")

    # --- TAB 2: multi-bp variant report ---
    with tab2:
        st.subheader("Multi-bp variant report (contiguous SUB/DEL/INS runs)")
        multi_bp_counts = {}
        for r in passed_reads:
            events = find_multi_bp_events(r["aligned_pairs"], ref_seq, r["query_seq"], min_len=min_event_length)
            for ev in events:
                key = (ev["start"], ev["end"], ev["type"], ev["ref_block"], ev["alt_block"])
                multi_bp_counts[key] = multi_bp_counts.get(key, 0) + 1

        multi_records = []
        for (start, end, ev_type, ref_block, alt_block), count in multi_bp_counts.items():
            total_depth = variant_counts[start]['Total'] if start in variant_counts else 0
            if total_depth == 0:
                continue
            freq = count / total_depth
            if freq >= variant_freq_threshold:
                multi_records.append({"Start_Position": start + 1, "End_Position": end + 1,
                                       "Length_bp": (end - start + 1) if ev_type != "INS" else len(alt_block),
                                       "Type": ev_type, "Reference_Bases": ref_block, "Mutant_Bases": alt_block,
                                       "Read_Count": count, "Total_Depth": total_depth, "Frequency": round(freq, 4)})
        df_multi = pd.DataFrame(multi_records)
        if not df_multi.empty:
            df_multi = df_multi.sort_values("Start_Position")
            st.dataframe(df_multi, use_container_width=True)
            st.download_button("Download CSV", df_multi.to_csv(index=False), "multi_bp_variant_report.csv")
        else:
            st.info("No multi-bp variants crossed the frequency threshold.")

    # --- TAB 3: CDR length distribution ---
    with tab3:
        st.subheader("CDR length distribution vs fitted binomial")
        if not cdr_regions:
            st.info("Set at least one CDR region in the sidebar (start/end must form whole codons).")
        else:
            for name, (s, e) in cdr_regions.items():
                s0, e0 = s - 1, e - 1
                ref_cdr_len = e - s + 1
                lengths = np.array([get_cdr_length(r["aligned_pairs"], s0, e0) for r in passed_reads])
                if lengths.max() == 0:
                    st.warning(f"{name}: no data.")
                    continue
                n_trials = int(lengths.max())
                p_hat = lengths.mean() / n_trials

                unique_lengths, obs_counts = np.unique(lengths, return_counts=True)
                obs_freq = obs_counts / obs_counts.sum()
                x_range = np.arange(0, n_trials + 1)
                pmf = binom.pmf(x_range, n_trials, p_hat)

                st.markdown(f"**{name}** — reference length {ref_cdr_len} bp, "
                            f"observed mean {lengths.mean():.2f} bp, fitted Binomial(n={n_trials}, p={p_hat:.3f})")

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(unique_lengths, obs_freq, width=0.8, alpha=0.6, color="steelblue", label="Observed")
                ax.plot(x_range, pmf, color="crimson", marker="o", markersize=3, label="Fitted binomial")
                ax.axvline(ref_cdr_len, color="black", linestyle="--", label="Reference length")
                ax.set_xlabel(f"{name} length (bp)")
                ax.set_ylabel("Frequency")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

    # --- TAB 4: CDR amino acid presence ---
    with tab4:
        st.subheader("CDR amino acid presence/absence vs fitted binomial")
        if not cdr_regions:
            st.info("Set at least one CDR region in the sidebar (start/end must form whole codons).")
        else:
            for name, (s, e) in cdr_regions.items():
                s0, e0 = s - 1, e - 1
                presence_lists = [get_amino_acid_presence(r["aligned_pairs"], ref_seq, r["query_seq"], s0, e0)
                                   for r in passed_reads]
                presence_arr = np.array(presence_lists)
                n_codons = presence_arr.shape[1]
                per_read_counts = presence_arr.sum(axis=1)
                p_hat = per_read_counts.mean() / n_codons

                st.markdown(f"**{name}** — {n_codons} amino acid positions, "
                            f"mean {per_read_counts.mean():.2f}/{n_codons} present, "
                            f"fitted Binomial(n={n_codons}, p={p_hat:.3f})")

                unique_counts, obs_counts = np.unique(per_read_counts, return_counts=True)
                obs_freq = obs_counts / obs_counts.sum()
                x_range = np.arange(0, n_codons + 1)
                pmf = binom.pmf(x_range, n_codons, p_hat)

                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.bar(unique_counts, obs_freq, width=0.8, alpha=0.6, color="seagreen", label="Observed")
                    ax.plot(x_range, pmf, color="crimson", marker="o", markersize=3, label="Fitted binomial")
                    ax.set_xlabel(f"# {name} amino acids present")
                    ax.set_ylabel("Frequency")
                    ax.legend()
                    st.pyplot(fig)
                    plt.close(fig)
                with col2:
                    position_rate = presence_arr.mean(axis=0)
                    fig2, ax2 = plt.subplots(figsize=(7, 4))
                    ax2.bar(np.arange(1, n_codons + 1), position_rate, color="steelblue")
                    ax2.set_ylim(0, 1.05)
                    ax2.set_xlabel(f"{name} position")
                    ax2.set_ylabel("Fraction present")
                    st.pyplot(fig2)
                    plt.close(fig2)
