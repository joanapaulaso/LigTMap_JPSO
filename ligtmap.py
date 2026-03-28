from __future__ import division

import csv
import os
import sys

if os.environ.get("LIGTMAP_DEBUG_SIM", "").lower() in ("1", "true", "yes"):
    print("DEBUG_SIM mode enabled")


def extract_smiles_from_line(line):
    stripped = line.strip()
    if not stripped:
        raise ValueError("Empty SMILES line")

    # Preferred: tab-separated CID, Name, SMILES
    if "\t" in stripped:
        parts = [field.strip().strip('"').strip("'") for field in stripped.split("\t")]
        if len(parts) >= 3:
            cid, name = parts[0], parts[1]
            smi = parts[-1]
            return cid, name, smi

    if "\t" in stripped:
        fields = [field.strip().strip('"').strip("'") for field in stripped.split("\t") if field.strip()]
        if fields:
            return None, None, fields[-1]

    if "," in stripped:
        fields = next(csv.reader([stripped], skipinitialspace=True))
        cleaned = [field.strip().strip('"').strip("'") for field in fields if field.strip()]
        if cleaned:
            return None, None, cleaned[-1]

    parts = stripped.split()
    if parts:
        return None, None, parts[-1].strip('"').strip("'")

    raise ValueError("Could not parse SMILES from line: {}".format(line))


def read_first_smiles(smile_path):
    with open(smile_path, "r") as handle:
        for raw_line in handle:
            if raw_line.strip():
                return extract_smiles_from_line(raw_line)
    raise ValueError("No SMILES found in {}".format(smile_path))


def main_part():
    smile_inf_name = sys.argv[1]
    input_num = sys.argv[2]
    tanifing_str = os.environ.get("LIGTMAP_TANIMOTO_CUTOFF", "0.85")
    try:
        tanifingcut = float(tanifing_str)
    except Exception:
        tanifingcut = 0.4
    select_db = sys.argv[3]
    total_db = int(sys.argv[4])
    rootpa = sys.argv[5]
    output_path = "Output/Input_" + input_num + "/" + select_db
    summary_path = "Output/Input_" + input_num

    cid, molname, smile = read_first_smiles(smile_inf_name)
    if cid is None:
        cid = os.path.splitext(os.path.basename(smile_inf_name))[0]
    if molname is None:
        molname = ""
    inp = smile
    output_list = []
    all_candidates = []
    fallback_top_n = int(os.environ.get("LIGTMAP_FALLBACK_TOPN", "20"))
    debug_errors = []

    output_root = "Output"
    os.makedirs(output_root, exist_ok=True)
    status_file_path = os.path.join(output_root, "status.txt")
    status_f = None
    try:
        status_f = open(status_file_path, "a")
        import rdkit  # noqa: F401
        import openbabel  # noqa: F401
        import pybel  # noqa: F401
        import oddt
        from rdkit import Chem
        from rdkit.Chem.EState import Fingerprinter
        from rdkit.Chem import Descriptors
        from rdkit.Chem.rdmolops import RDKFingerprint
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import StandardScaler

        # Prefer native pychem if importable; otherwise provide a minimal rdkit-backed shim.
        try:
            import pychem  # type: ignore
            from pychem.pychem import Chem as pychem_Chem  # type: ignore
            from pychem import fingerprint as pychem_fingerprint  # type: ignore
            PYCH_AVAILABLE = True
        except Exception:
            PYCH_AVAILABLE = False

            class _ShimChem:
                @staticmethod
                def MolFromSmiles(smi):
                    return Chem.MolFromSmiles(smi)

            class _ShimFingerprint:
                @staticmethod
                def CalculateMorganFingerprint(mol, radius=2):
                    fp = Chem.AllChem.GetMorganFingerprint(mol, radius)
                    return (None, None, fp)

                @staticmethod
                def CalculateMACCSFingerprint(mol):
                    from rdkit.Chem import MACCSkeys

                    fp = MACCSkeys.GenMACCSKeys(mol)
                    return (None, None, fp)

                @staticmethod
                def CalculateDaylightFingerprint(mol):
                    fp = Chem.RDKFingerprint(mol)
                    return (None, None, fp)

                @staticmethod
                def CalculateSimilarity(fp1, fp2, metric="Tanimoto"):
                    # Use appropriate similarity depending on fingerprint types
                    if isinstance(fp1, rdkit.DataStructs.cDataStructs.UIntSparseIntVect) or isinstance(
                        fp2, rdkit.DataStructs.cDataStructs.UIntSparseIntVect
                    ):
                        return rdkit.DataStructs.TanimotoSimilarity(fp1, fp2)
                    return Chem.DataStructs.FingerprintSimilarity(fp1, fp2)

            pychem_Chem = _ShimChem
            pychem_fingerprint = _ShimFingerprint

        from rdkit import Chem, DataStructs, RDConfig
        from rdkit.Chem import AllChem
        from rdkit.Chem import ChemicalFeatures
        from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
        from rdkit.Chem.rdMolDescriptors import GetHashedAtomPairFingerprintAsBitVect, GetHashedTopologicalTorsionFingerprintAsBitVect
        from rdkit.Chem.AtomPairs.Sheridan import GetBPFingerprint
        from rdkit.Chem.EState.Fingerprinter import FingerprintMol
        from rdkit.Avalon.pyAvalonTools import GetAvalonFP  # GetAvalonCountFP  #int vector version
        from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect, GetErGFingerprint
        from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
        import rdkit.DataStructs.cDataStructs
        from sklearn.metrics import r2_score
        from numpy.random import randn
        from numpy.random import seed
        from scipy.stats import pearsonr
        import csv
        import subprocess
        import pickle
        import copy

        from six.moves import zip_longest
        from itertools import chain
        from collections import OrderedDict
        import numpy as np
        from scipy.sparse import csr_matrix, isspmatrix_csr
        import oddt
        from oddt.utils import is_openbabel_molecule
        from oddt.interactions import (
            pi_stacking,
            hbond_acceptor_donor,
            hbonds,
            salt_bridge_plus_minus,
            hydrophobic_contacts,
            acceptor_metal,
            close_contacts,
        )
        from oddt import toolkit
        from oddt import shape
        from oddt import fingerprints
        from rdkit.Chem import Draw
        from oddt import interactions

        __all__ = [
            "InteractionFingerprint",
            "SimpleInteractionFingerprint",
            "SPLIF",
            "similarity_SPLIF",
            "ECFP",
            "PLEC",
            "dice",
            "tanimoto",
            "close_contacts",
            "hbond_acceptor_donor",
            "hbonds",
            "halogenbond_acceptor_halogen",
            "halogenbonds",
            "pi_stacking",
            "salt_bridge_plus_minus",
            "salt_bridges",
            "hydrophobic_contacts",
            "pi_cation",
            "acceptor_metal",
            "pi_metal",
        ]

        print("Step 1: Ligand Similarity Search")
        # PART1: Similarity <--> Tanifing + Tanipharm Score

        try:
            # Generate the fingerprint
            # factory = Gobbi_Pharm2D.factory
            mol1 = pychem_Chem.MolFromSmiles(inp)
            AllChem.EmbedMolecule(mol1, useRandomCoords=True)
            # Precompute fingerprints with RDKit for compatibility
            FP_inp_Morgan = AllChem.GetMorganFingerprint(mol1, 2)
            from rdkit.Chem import MACCSkeys

            FP_inp_MACCSF = MACCSkeys.GenMACCSKeys(mol1)
            FP_inp_Daylight = RDKFingerprint(mol1)
        except Exception:
            with open(summary_path + "/smile_err.dat", "w"):
                pass
        if os.environ.get("LIGTMAP_DEBUG_SIM", "").lower() in ("1", "true", "yes"):
            print("DEBUG_SIM fingerprints ready")
        # --- Similarity search using RDKit fingerprints ---
        id_ligname = {}
        id_proname = {}
        index_processed = 0
        index_path = os.path.join(rootpa, "Index", select_db + ".csv")
        with open(index_path, "r") as index_file:
            label_line = index_file.readline()
            for y in index_file:
                index_processed += 1
                y = y.rstrip()
                parts = y.split(";")
                if len(parts) < 5:
                    continue
                data_pdb, data_smile, affinity = parts[0], parts[1], parts[2]
                id_proname[data_pdb] = parts[3]
                id_ligname[data_pdb] = parts[4]
                # Load stored fingerprints; skip if missing
                morgan_path = os.path.join(rootpa, "Index/fing/Morgan", select_db, data_pdb + ".bin")
                maccs_path = os.path.join(rootpa, "Index/fing/MACCSF/New", select_db, data_pdb + ".dat")
                if not (os.path.isfile(morgan_path) and os.path.isfile(maccs_path)):
                    debug_errors.append("MISSING_FP:%s" % data_pdb)
                    continue

                with open(morgan_path, "rb") as data_morgan_file:
                    data_morgan_bin = data_morgan_file.read()
                    data_morgan = DataStructs.UIntSparseIntVect(data_morgan_bin)

                with open(maccs_path, "r", encoding="utf-8", errors="ignore") as data_maccsf_file:
                    data_maccsf_bin = data_maccsf_file.readline().rstrip()
                    data_maccsf = DataStructs.CreateFromBitString(data_maccsf_bin)

                Tanifing_Morgan = DataStructs.TanimotoSimilarity(FP_inp_Morgan, data_morgan)
                Tanifing_MACCSF = DataStructs.FingerprintSimilarity(FP_inp_MACCSF, data_maccsf)

                if total_db > 1:
                    with open(os.path.join(rootpa, "Index/fing/Daylight", select_db, data_pdb + ".dat"), "r", encoding="utf-8", errors="ignore") as data_daylight_file:
                        data_daylight_bin = data_daylight_file.readline().rstrip()
                        data_daylight = DataStructs.CreateFromBitString(data_daylight_bin)
                    Tanifing_Daylight = DataStructs.FingerprintSimilarity(FP_inp_Daylight, data_daylight)
                    sum_score = (Tanifing_Morgan + Tanifing_MACCSF + Tanifing_Daylight) / 3
                else:
                    sum_score = (Tanifing_Morgan + Tanifing_MACCSF) / 2

                candidate_row = [
                    inp,
                    data_smile,
                    data_pdb,
                    affinity,
                    str(Tanifing_Morgan),
                    str(Tanifing_MACCSF),
                    str(sum_score),
                ]
                all_candidates.append((sum_score, candidate_row))
                if sum_score >= tanifingcut:
                    output_list.append(candidate_row)

        # PART1 COMPLETED
        print("DEBUG summary raw counts: candidates=%d above_cutoff=%d errors=%d" % (len(all_candidates), len(output_list), len(debug_errors)))
        if all_candidates:
            all_candidates_sorted = sorted(all_candidates, key=lambda s: s[0], reverse=True)
            top_score = all_candidates_sorted[0][0]
            top_ids = [row[2] for _, row in all_candidates_sorted[:5]]
        else:
            all_candidates_sorted = []
            top_score = "n/a"
            top_ids = []

        print(
            "Similarity summary: candidates=%d above_cutoff=%d cutoff=%s top_score=%s top_ids=%s processed=%d errors=%d"
            % (len(all_candidates), len(output_list), tanifingcut, top_score, ",".join(top_ids), index_processed, len(debug_errors))
        )
        print("DEBUG_SIM after summary print")
        # Persist debug info per target for inspection
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "similarity_debug.txt"), "w") as dbg:
            dbg.write(
                "candidates=%d above_cutoff=%d cutoff=%s top_score=%s top_ids=%s\n"
                % (len(all_candidates), len(output_list), tanifingcut, top_score, ",".join(top_ids))
            )
        if debug_errors:
            print("Encountered %d errors in similarity step, see error.txt" % len(debug_errors))
            with open(os.path.join(output_path, "error.txt"), "a") as error_file:
                for err in debug_errors[:5]:
                    error_file.write("EXC:%s\n" % err)

        # Always cap to top-N to avoid huge docking runs and ensure we keep highest scores
        if all_candidates and fallback_top_n > 0:
            all_candidates.sort(key=lambda s: s[0], reverse=True)
            output_list = [row for _, row in all_candidates[:fallback_top_n]]

        if not output_list:
            # No ligands passed similarity cutoff; write placeholder outputs and exit gracefully.
            os.makedirs(output_path, exist_ok=True)
            out_csv = os.path.join(output_path, "output.csv")
            with open(out_csv, "w") as out_file:
                out_file.write("Input;Data;PDB;Affinity;Tani_Morgan;Tani_MACCSF;LigandScore;ILDScore\n")
            status_f.write(input_num + ":" + select_db + ":NoHits\n")
            status_f.close()
            print(
                "No candidates passed similarity cutoff; skipping docking/prediction. "
                "all_candidates=%d, cutoff=%s, top_score=%s"
                % (
                    len(all_candidates),
                    tanifingcut,
                    all_candidates[0][0] if all_candidates else "n/a",
                )
            )
            return

        print("Step 2: Docking")
        # PART2: Docking
        docking_ok = False
        try:
            # Prepare the input ligand
            command = ["obabel", "-:" + smile, "-opdb", "-O", output_path + "/input.pdb", "--gen3d"]
            try:
                subprocess.check_output(command, stderr=subprocess.STDOUT)
            except Exception:
                # fallback to RDKit 3D generation
                try:
                    mol = Chem.AddHs(Chem.MolFromSmiles(smile))
                    AllChem.EmbedMolecule(mol, useRandomCoords=True)
                    AllChem.UFFOptimizeMolecule(mol)
                    Chem.MolToPDBFile(mol, output_path + "/input.pdb")
                except Exception as rd_exc:
                    raise rd_exc

            # prepare ligand for docking
            command = [
                os.environ.get("MGLTools") + "/bin/pythonsh",
                os.environ.get("MGLTools") + "/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py",
                "-l",
                output_path + "/input.pdb",
                "-o",
                output_path + "/input.pdbqt",
                "-A",
                "hydrogens",
                "-U",
                "nphs_lps_waters",
            ]
            subprocess.check_output(command, stderr=subprocess.STDOUT)

            # Get the PDBID
            pid_list = []
            with open(output_path + "/file_list", "w") as out_file:
                for ele in output_list:
                    out_file.write(ele[2] + "\n")
                    pid_list.append(ele[2])

            # Docking
            command = [rootpa + "docking_files/dock.sh", "pso", output_path, "1", "1", rootpa]
            subprocess.check_call(command, stderr=subprocess.STDOUT)
            docking_ok = True
        except Exception as dock_exc:
            docking_ok = False
            os.makedirs(output_path, exist_ok=True)
            with open(os.path.join(output_path, "docking_error.txt"), "a") as df:
                df.write(str(dock_exc) + "\n")
            print("Docking failed, skipping docking-dependent steps: {}".format(dock_exc))

        # Write the docking score to the output.csv
        if docking_ok:
            with open(output_path + "/DOCK_LOG/score_1.dat", "r") as score_file:
                score_list = score_file.readlines()

            with open(output_path + "/output.csv", "w") as out_file:
                out_file.write("CID;Name;Input;Data;PDB;Affinity;Tani_Morgan;Tani_MACCSF;LigandScore;ILDScore\n")
                i = 0
                for line in output_list:
                    out_line = ";".join([cid, molname] + line)
                    idscore = score_list[i].rstrip()
                    if idscore != "":
                        out_line = out_line + ";" + idscore + "\n"
                    else:
                        out_line = out_line + ";n.a.\n"
                    out_file.write(out_line)
                    i = i + 1
        else:
            # docking failed: write output with zero dock scores
            with open(output_path + "/output.csv", "w") as out_file:
                out_file.write("CID;Name;Input;Data;PDB;Affinity;Tani_Morgan;Tani_MACCSF;LigandScore;ILDScore\n")
                for line in output_list:
                    out_line = ";".join([cid, molname] + line)
                    out_line = out_line + ";0.0\n"
                    out_file.write(out_line)

        # Create complex file if docking succeeded
        if docking_ok:
            os.makedirs(output_path + "/Complex", exist_ok=True)
            for pdbid in pid_list:
                ligand_filename = output_path + "/DOCK_LOG/" + pdbid + "/" + pdbid + "_ligand_1.pdb"
                protein_filename = rootpa + "docking_files/protein/" + pdbid + "_protein.pdb"
                filenames = [protein_filename, ligand_filename]
                output_filename = output_path + "/Complex/complex_" + pdbid + ".pdb"
                with open(output_filename, "w") as outfile:
                    for fname in filenames:
                        if os.path.isfile(fname):
                            with open(fname, "r") as infile:
                                outfile.write(infile.read())

        print("Step 3: Activity Prediction")
        # PART3: Activity prediction
        # Read the data, affinity prediction
        csv_name = output_path + "/output.csv"
        data = pd.read_csv(csv_name, sep=";", header=[0], encoding="latin1")

        # Add some new columns
        data["Mol"] = data["Input"].apply(Chem.MolFromSmiles)
        num_mols = len(data)

        def MorganFingerprint(mol):
            return FingerprintMol(mol)[0]

        # Scale X to unit variance and zero mean
        data["Fingerprint"] = data["Mol"].apply(MorganFingerprint)

        X = np.array(list(data["Fingerprint"]))

        st = StandardScaler()
        X = np.array(list(data["Fingerprint"]))
        Test = X

        def ExplicitBitVect_to_NumpyArray(bitvector):
            bitstring = bitvector.ToBitString()
            intmap = map(int, bitstring)
            return np.array(list(intmap))

        class fingerprint:
            def __init__(self, fp_fun, name):
                self.fp_fun = fp_fun
                self.name = name
                self.x = []

            def apply_fp(self, mols):
                for mol in mols:
                    fp = self.fp_fun(mol)
                    if isinstance(fp, tuple):
                        fp = np.array(list(fp[0]))
                    if isinstance(fp, rdkit.DataStructs.cDataStructs.ExplicitBitVect):
                        fp = ExplicitBitVect_to_NumpyArray(fp)
                    if isinstance(fp, rdkit.DataStructs.cDataStructs.IntSparseIntVect):
                        fp = np.array(list(fp))

                    self.x += [fp]

                    if str(type(self.x[0])) != "<class 'numpy.ndarray'>":
                        print("WARNING: type for ", self.name, "is ", type(self.x[0]))

        # Load the model
        with open(rootpa + "Model/" + select_db + ".sav", "rb") as f:
            try:
                Model = pickle.load(f, encoding="latin1")
            except TypeError:
                Model = pickle.load(f)

        predictions = Model.predict(Test)

        # PART3 COMPLETED

        if not globals().get("docking_ok", True):
            status_f.write(input_num + ":" + select_db + ":NoDock\n")
            status_f.close()
            print("Skipping binding similarity because docking failed.")
            return

        countline = -1
        with open(output_path + "/output.csv", "r") as countlinefile:
            for line in countlinefile:
                countline += 1

        print("Step 4: Binding Similarity Search (Total " + str(countline) + " target pdb, please wait...)")
        # PART4: Similarity <--> Binding fingerprint

        def tanimoto(a, b, sparse=False):
            if sparse:
                a = np.unique(a)
                b = np.unique(b)
                a_b = float(len(np.intersect1d(a, b, assume_unique=True)))
                denominator = len(a) + len(b) - a_b
                if denominator > 0:
                    return a_b / denominator
            else:
                a = a.astype(bool)
                b = b.astype(bool)
                a_b = (a & b).sum().astype(float)
                denominator = a.sum() + b.sum() - a_b
                if denominator > 0:
                    return a_b / denominator
            return 0.0

        IFP_filename = summary_path + "/IFP_result.csv"
        with open(output_path + "/output.csv", "r") as first_result_table:
            ignore_label = first_result_table.readline()
            for line in first_result_table:
                line_list = line.rstrip().split(";")
                if len(line_list) < 10:
                    continue
                # output.csv header: CID;Name;Input;Data;PDB;Affinity;Tani_Morgan;Tani_MACCSF;LigandScore;ILDScore
                r_cid = line_list[0]
                r_name = line_list[1]
                r_smiles = line_list[2] if len(line_list) > 2 else ""
                r_pdbid = line_list[4]
                r_ligandscore = line_list[8]
                r_dockscore = line_list[9]

                bindscore = None
                if os.path.isfile(rootpa + "Index/IFP/" + select_db + "/" + r_pdbid + ".bin") and os.path.isfile(
                    output_path + "/DOCK_LOG/" + r_pdbid + "/" + r_pdbid + "_ligand_1.pdb"
                ):
                    crystal_IFP = np.fromfile(rootpa + "Index/IFP/" + select_db + "/" + r_pdbid + ".bin", dtype=np.uint8)
                    bind_protein = next(oddt.toolkit.readfile("pdb", rootpa + "docking_files/protein/" + r_pdbid + "_protein.pdb"))
                    bind_protein.protein = True
                    bind_ligand = next(oddt.toolkit.readfile("pdb", output_path + "/DOCK_LOG/" + r_pdbid + "/" + r_pdbid + "_ligand_1.pdb"))
                    IFP = fingerprints.InteractionFingerprint(bind_ligand, bind_protein)
                    if crystal_IFP.shape == IFP.shape:
                        bindscore = tanimoto(crystal_IFP, IFP)

                if bindscore is not None:
                    ligtmapscore = 0.7 * float(r_ligandscore) + 0.3 * bindscore
                else:
                    ligtmapscore = float(r_ligandscore)

                # Write the binding fingerprint score to the IFP_result.csv (sort)
                if not os.path.isfile(IFP_filename):
                    with open(IFP_filename, "w") as IFP_file:
                        IFP_file.write(
                            "CID;Name;SMILES;PDB;Class;TargetName;LigandName;LigandSimilarityScore;BindingSimilarityScore;LigTMapScore;PredictedAffinity;DockingScore\n"
                        )
                        IFP_file.write(
                            r_cid
                            + ";"
                            + r_name
                            + ";"
                            + r_smiles
                            + ";"
                            + r_pdbid
                            + ";"
                            + select_db
                            + ";"
                            + id_proname[r_pdbid]
                            + ";"
                            + id_ligname[r_pdbid]
                            + ";"
                            + str(round(float(r_ligandscore), 6))
                            + ";"
                            + (str(round(bindscore, 6)) if bindscore is not None else "n.a.")
                            + ";"
                            + str(round(ligtmapscore, 6))
                            + ";"
                            + str(round(predictions[0], 6))
                            + ";"
                            + str(round(float(r_dockscore), 6))
                            + "\n"
                        )
                else:
                    tmp_list = []
                    with open(IFP_filename, "r") as tmp_file:
                        label_line = tmp_file.readline()
                        for line in tmp_file.readlines():
                            line = line.rstrip()
                            line = line.split(";")
                            tmp_list.append(line)
                        tmp_list.append(
                            [
                                r_cid,
                                r_name,
                                r_smiles,
                                r_pdbid,
                                select_db,
                                id_proname[r_pdbid],
                                id_ligname[r_pdbid],
                                str(round(float(r_ligandscore), 6)),
                                str(round(bindscore, 6)) if bindscore is not None else "n.a.",
                                str(round(ligtmapscore, 6)),
                                str(round(predictions[0], 6)),
                                str(round(float(r_dockscore), 6)),
                            ]
                        )
                    tmp_list.sort(key=lambda s: float(s[9]), reverse=True)
                    with open(IFP_filename, "w") as IFP_file:
                        IFP_file.write(label_line)
                        for ele in tmp_list:
                            write_str = ";".join(ele)
                            IFP_file.write(write_str + "\n")
                # missing IFP handled above; nothing to append here
        # PART4 COMPLETED

        status_f.write(input_num + ":" + select_db + ":Complete\n")
        status_f.close()
        print("DONE")
    except Exception as e:
        status_f = status_f or open(status_file_path, "a")
        status_f.write(input_num + ":" + select_db + ":Fail\n")
        status_f.close()
        print("NO RESULT: {}".format(e))


main_part()
