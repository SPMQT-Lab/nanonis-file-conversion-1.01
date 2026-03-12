#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: rnpla, @coauthor: GCampi
dat_sxm_cli.py

Run with defaults:
    python dat_sxm_cli.py

Run with CLI args:
    python dat_sxm_cli.py --input-dir "C:/path/to/dat_or_folder" --output-dir "C:/path/to/out"
    python dat_sxm_cli.py --input-dir "C:/path/to/dat_or_folder" --output-dir "C:/path/to/out" --cushion-dir "C:/path/to/file_cushions"
"""

from pathlib import Path
import argparse
import numpy as np
import re
import shutil
import zlib
import math
import json
import os
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

# ═════════════════════════════════════════════════════════════════════════════
# ════════════ P R E L I M I N A R Y ══════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════════════

# ———————————— PARAMETERS_DEFAULTS ————————————

REPO_ROOT = Path(__file__).resolve().parent.parent

#########################################################################################
############### YOU CAN AND SHOULD CHANGE THESE TO SUIT YOUR NEEDS ######################
#########################################################################################

DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "sample_input"

DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "output_sxm"

DEFAULT_CUSHION_DIR = REPO_ROOT / "src" / "file_cushions"

# ———LEAVE THE BELOW.—————————————————————————————————————————————————————————————————————
# below are Some archaic conventions I haven’t fixed yet :) 
dat_source = DEFAULT_INPUT_DIR
rebuilt_path = DEFAULT_OUTPUT_DIR
cushion_path = DEFAULT_CUSHION_DIR
out_dir = rebuilt_path

# ———————————— TUNERS ————————————


delete_all_content = 0 # delete all the content of the folder true 1 false 0
delete_files = () # list the files you want deleted. 0,1,2,8 as an example.

# ignore the below tuners, these arnt currently implemented for your version.
base_names = ('4', '5')
force_name = ''
force_active_base = -1
num_files_p_base = 9

# ———————————— VARIABLES ————————————

percentile: np.ndarray = np.array([1, 99])
data_dac_conversion_factor: float = 9.536 * 10 ** (-6)

default_scaling_parameters: Dict[str, float] = {
    'Num.X': 512,
    'Num.Y': 512,
    'GainX': 10.0,
    'GainY': 10.0,
    'GainZ': 10.0,
    'XPiezoconst': 96.0,
    'YPiezoconst': 96.0,
    'ZPiezoconst': 19.2,
    'PreAmpSensi': 10**9
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert Nanonis .dat -> .sxm")
    p.add_argument(
        "legacy_input",
        nargs="?",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "legacy_output",
        nargs="?",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "legacy_cushion",
        nargs="?",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--input-dir",
        dest="input_dir",
        default=None,
        help="Path to a .dat file or a directory containing .dat files",
    )
    p.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Output directory for .sxm files",
    )
    p.add_argument(
        "--cushion-dir",
        dest="cushion_dir",
        default=None,
        help="Directory containing file cushion layout files",
    )
    return p.parse_args()

# ———————————— FILE DELETING AND NAMING TOOLS ————————————

def dc(folder_path: Path) -> None:
    """CLEAR WHOLE FOLDER (contents only)"""
    folder_path = Path(folder_path)
    if not folder_path.exists():
        return
    for item in folder_path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    print(f"Contents of '{folder_path}' have been cleared.")

def naming(
    folder_path: Path = None,
    bases=base_names,
    max_num=num_files_p_base-1,
    extension='.sxm',
    force_active=force_active_base
) -> str:
    folder_path = Path(folder_path) if folder_path is not None else rebuilt_path
    folder_path.mkdir(parents=True, exist_ok=True)

    # 1) Determine active base
    if force_active == -1:
        alt_base_found = any(
            item.stem.startswith(bases[1])
            for item in folder_path.iterdir()
            if item.is_file()
        )
        active_base = bases[1] if alt_base_found else bases[0]
    else:
        active_base = bases[force_active]

    final_name = f"{active_base}{max_num}_FINAL{extension}"
    final_path = folder_path / final_name
    if final_path.exists():
        active_base = list(set(bases) - set([active_base]))[0]

    # 2) Find highest existing number for active base
    max_found_num = -1
    pattern = re.compile(rf"^{re.escape(active_base)}(\d+)$")  # CAPTURE THE NUMBER

    for item in folder_path.iterdir():
        if item.is_file() and item.suffix == extension:
            m = pattern.match(item.stem)
            if m:
                num_str = m.group(1)
                try:
                    max_found_num = max(max_found_num, int(num_str))
                except ValueError:
                    continue

    # 3) Next filename
    next_num = max_found_num + 1
    if next_num < max_num:
        return f"{active_base}{next_num}{extension}"
    else:
        return final_name

def del_files(
    *indices,
    folder_path: Path = None,
    bases=('test', 'alt_test'),
    num_files_p_base=9,
    extension='.sxm'
) -> None:
    folder_path = Path(folder_path) if folder_path is not None else rebuilt_path
    if not folder_path.exists():
        return

    full_bases = []
    for base in bases:
        if (folder_path / f"{base}{num_files_p_base}_FINAL{extension}").exists():
            full_bases.append(base)

    if not full_bases:
        print("No full bases found. Exiting without changes.")
        return

    for base in full_bases:
        print(f"\n--- Processing full base '{base}' ---")

        files_to_delete = []
        for i in indices:
            if 0 <= i < num_files_p_base:
                filename = f"{base}{i}{extension}"
            else:
                print(f"Index {i} is out of range (0-{num_files_p_base - 1}). Skipping.")
                continue

            filepath = folder_path / filename
            if filepath.exists():
                files_to_delete.append(filepath)

        # 1) Delete
        for file_to_del in files_to_delete:
            try:
                os.unlink(file_to_del)
                print(f"Deleted: {file_to_del.name}")
            except OSError as e:
                print(f"Error deleting file {file_to_del.name}: {e}")

        # 2) Remaining files (sorted)
        remaining_files = sorted(
            [
                item for item in folder_path.iterdir()
                if item.is_file() and item.name.startswith(base) and item.suffix == extension
            ],
            key=lambda x: int(re.search(r'\d+', x.stem).group()) if re.search(r'\d+', x.stem) else -1
        )

        final_file = folder_path / f"{base}{num_files_p_base}_FINAL{extension}"
        if final_file in remaining_files:
            remaining_files.remove(final_file)
            remaining_files.append(final_file)

        # 3) Rename sequence
        new_names = [f"{base}{i}{extension}" for i in range(len(remaining_files))]

        print("\nRenaming remaining files...")
        for old_path, new_name in zip(remaining_files, new_names):
            if old_path.name != new_name:
                new_path = old_path.parent / new_name
                old_path.rename(new_path)
                print(f"Renamed '{old_path.name}' to '{new_path.name}'")

def apply_tuners() -> None:
    """Apply delete/naming tuners AFTER args set globals."""
    global name_of_built_file

    rebuilt_path.mkdir(parents=True, exist_ok=True)

    if delete_all_content == 1:
        dc(rebuilt_path)

    if force_name != '':
        name_of_built_file = force_name
    else:
        name_of_built_file = naming(folder_path=rebuilt_path)

    if delete_files != ():
        del_files(*delete_files, folder_path=rebuilt_path)

# ———————————— MOST GENERAL TOOLS ————————————

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _nums(txt: str, n=None):
    xs = [float(t) for t in re.findall(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', txt or "")]
    return xs if n is None else xs[:n]

def _E(x: float, prec: int) -> str:
    s = f"{float(x):.{prec}E}"
    return re.sub(r'E([+-])0?(\d)(?!\d)', r'E\1\2', s)

def to_f32(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32)

# ———————————— IMAGE STACK MANIPULATION TOOLS ————————————

def equalise(data: np.ndarray) -> np.ndarray:
    return data - np.min(data)

def find_last_valid_point(data: np.ndarray) -> List[int]:
    valid_mask = np.logical_and(~np.isnan(data), data != 0)
    rows, cols = np.where(valid_mask)
    if len(rows) == 0:
        return [0, 0]
    max_row = int(np.max(rows))
    max_col_in_max_row = int(np.max(cols[rows == max_row]))
    return [max_row, max_col_in_max_row]

def trim(stack: np.ndarray) -> List[Any]:
    if stack is None or stack.size == 0:
        return [np.array([])]
    channel0_data = stack[0]
    num_rows, num_cols = channel0_data.shape
    last_row_ch0, last_col_ch0 = find_last_valid_point(channel0_data)

    is_on_right_edge = (last_col_ch0 == num_cols - 1)
    new_num_rows = (last_row_ch0 + 1) if is_on_right_edge else last_row_ch0
    new_num_rows = max(0, int(new_num_rows))

    trimmed_channels: List[np.ndarray] = []
    for i in range(stack.shape[0]):
        trimmed_channels.append(stack[i][:new_num_rows, :])
    return [np.array(trimmed_channels), new_num_rows]

# ———————————— LOADERS ————————————

def load_sxm_layout(cushion_path_in: Path) -> dict:
    cushion_path_in = Path(cushion_path_in)

    files = {
        "post_end_bytes": cushion_path_in / "post_end_bytes.bin",
        "pre_payload_bytes": cushion_path_in / "pre_payload_bytes.bin",
        "pad_len": cushion_path_in / "pad_len.txt",
        "data_offset": cushion_path_in / "data_offset.txt",
        "tail_bytes": cushion_path_in / "tail_bytes.bin",
    }
    missing = [k for k, p in files.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing layout files: {', '.join(missing)} in {cushion_path_in}")

    post_end_bytes = files["post_end_bytes"].read_bytes()
    pre_payload_bytes = files["pre_payload_bytes"].read_bytes()
    pad_len = int(files["pad_len"].read_text(encoding="utf-8").strip())
    data_offset = int(files["data_offset"].read_text(encoding="utf-8").strip())
    tail_bytes = files["tail_bytes"].read_bytes()

    if len(pre_payload_bytes) != pad_len:
        raise ValueError(
            f"pad_len ({pad_len}) != len(pre_payload_bytes) ({len(pre_payload_bytes)}). Re-run boundary capture."
        )

    return {
        "post_end_bytes": post_end_bytes,
        "pre_payload_bytes": pre_payload_bytes,
        "pad_len": pad_len,
        "data_offset": data_offset,
        "tail_bytes": tail_bytes,
    }

def load_header_format(fmt_path: str | Path) -> dict:
    fmt_path = Path(fmt_path)
    obj = json.loads(fmt_path.read_text(encoding="utf-8"))

    required = ["marker", "line_ending", "block_order", "between_blocks", "key_case", "filler"]
    missing = [k for k in required if k not in obj]
    if missing:
        raise ValueError(f"header_format.json missing keys: {', '.join(missing)}")

    le = obj["line_ending"]
    if le in ("\\r\\n", r"\r\n"):
        obj["line_ending"] = "\r\n"
    elif le in ("\\n", r"\n"):
        obj["line_ending"] = "\n"

    obj.setdefault("numeric_hints", {})
    obj.setdefault("data_info", {"columns": [], "delimiter": "\\t", "include_header_row": True})
    obj["marker"].setdefault("newline_after_marker", True)

    defaults = {
        "REC_TEMP_LPAD": 6, "ACQ_LPAD": 7,
        "PIX_LPAD": 7, "PIX_SEP": 7,
        "E3_LPAD": 13, "E3_SEP": 13,
        "E6_LPAD": 11, "E6_SEP": 11,
        "OFF_LPAD": 13, "OFF_SEP": 9,
        "ANGLE_LPAD": 12, "BIAS_LPAD": 12,
        "TYPE1": 14, "TYPE2": 12,
        "DATE_LPAD": 1,
    }
    obj["left_pads"] = {**defaults, **obj.get("left_pads", {})}
    return obj

def load() -> Tuple[dict, dict]:
    layout = load_sxm_layout(cushion_path)
    header_format_path = Path(cushion_path) / "header_format.json"
    header_format = load_header_format(header_format_path)
    return layout, header_format

# ═════════════════════════════════════════════════════════════════════════════
# ════════════ H E A D E R  P R O C E S S I N G ═══════════════════════════════
# ═════════════════════════════════════════════════════════════════════════════

def parse_header(hb: bytes) -> Dict[str, str]:
    hdr: Dict[str, str] = {}
    for line in hb.splitlines():
        if b'=' in line:
            k, v = line.split(b'=', 1)
            key = k.decode('ascii', errors='ignore').split('/')[-1].strip()
            val = v.decode('ascii', errors='ignore').strip()
            hdr[key] = val
    return hdr

def find_hdr(hdr: Dict[str, str], hint: str, default: Optional[Any] = None) -> str:
    for k in hdr:
        if hint.lower() in k.lower():
            return hdr[k]
    if default is not None:
        return default
    raise KeyError(f"Key not found in header: {hint}")

def parse_dat_timestamp(fname: str) -> datetime:
    name = Path(fname).name
    return datetime.strptime(name, "A%y%m%d.%H%M%S.dat")

def construct_hdr(dat_hdr: Dict[str, str], dat_path: Path, numChan: int) -> Dict[str, str]:
    def sci_fixed_dec(x, dec=6):
        s = f"{float(x):.{dec}E}".upper()
        return re.sub(r"E([+-])0?(\d+)$", lambda m: f"E{m.group(1)}{int(m.group(2))}", s)

    dt = parse_dat_timestamp(dat_path.name)
    date = dt.strftime("%d.%m.%Y")
    time = dt.strftime("%H:%M:%S")
    file_str = str(dat_path)

    temperature = dat_hdr["T_AUXADC6[K]"]

    numx = dat_hdr["Num.X"]
    numy = dat_hdr["Num.Y"]
    lengthx_angstroms = float(dat_hdr["Length x[A]"])
    lengthy_angstroms = float(dat_hdr["Length y[A]"])
    lengthx_m = sci_fixed_dec(lengthx_angstroms * 1e-10, 6)
    lengthy_m = sci_fixed_dec(lengthy_angstroms * 1e-10, 6)

    total_time_duration = dat_hdr["Image:"]
    duration_line = float(dat_hdr["line:"])
    duration_line_sci_notation = sci_fixed_dec(duration_line, 3)

    offsetx_dac = float(dat_hdr["OffsetX"])
    offsety_dac = float(dat_hdr["OffsetY"])
    dactoAxy = float(dat_hdr["Dacto[A]xy"])
    offsetx_m = sci_fixed_dec(offsetx_dac * dactoAxy * 1e-10, 6)
    offsety_m = sci_fixed_dec(offsety_dac * dactoAxy * 1e-10, 6)

    bias_mV = float(str(dat_hdr["Biasvolt[mV]"]))
    bias_V = bias_mV * 1e-3
    bias_str = sci_fixed_dec(bias_V, 3)

    scan_dir_str = "down" if str(dat_hdr.get("ScanYDirec", "1")).strip() == "1" else "up"
    angle_rad = math.radians(float(str(dat_hdr.get("Rotation", "0")).replace(",", ".")))
    angle_str = sci_fixed_dec(angle_rad, 3)

    comment_str = (dat_hdr.get("Titel", "") or "Empty").strip()

    z_name = "log Current" if str(dat_hdr.get("FBLog", "0")).strip() == "1" else "Current"
    z_on = "0" if str(dat_hdr.get("FBOff", "0")).strip() == "1" else "1"
    setp = float(str(dat_hdr.get("SetPoint", "0")).replace(",", "."))
    setp_str = sci_fixed_dec(setp, 3) + " A"

    hdr: Dict[str, str] = {}

    hdr["NANONIS_VERSION"] = "2"
    hdr["SCANIT_TYPE"] = "FLOAT            MSBFIRST"

    hdr["REC_DATE"] = date
    hdr["REC_TIME"] = time

    hdr["REC_TEMP"] = str(temperature)
    hdr["ACQ_TIME"] = str(total_time_duration)

    hdr["SCAN_PIXELS"] = f"{numx}       {numy}"
    hdr["SCAN_FILE"] = file_str

    hdr["SCAN_TIME"] = f"{duration_line_sci_notation}             {duration_line_sci_notation}"
    hdr["SCAN_RANGE"] = f"{lengthx_m}           {lengthy_m}"
    hdr["SCAN_OFFSET"] = f"{offsetx_m}         {offsety_m}"

    hdr["SCAN_ANGLE"] = angle_str
    hdr["SCAN_DIR"] = scan_dir_str
    hdr["BIAS"] = bias_str

    hdr["Z-CONTROLLER"] = (
        "Name\ton\tSetpoint\tP-gain\tI-gain\tT-const\n"
        f"\t{z_name}\t{z_on}\t{setp_str}\t1.000E+0 m\t1.000E+0 m/s\t0.000E+0 s"
    )

    hdr["COMMENT"] = comment_str

    hdr["DATA_INFO"] = (
        "Channel\tName\tUnit\tDirection\tCalibration\tOffset\n"
        "\t14\tZ\tm\tboth\t1.000E+0\t0.000E+0\n"
        "\t0\tCurrent\tA\tboth\t1.000E+0\t0.000E+0"
    )

    hdr["Clip_percentile_Lower"] = str(percentile[0])
    hdr["Clip_percentile_higher"] = str(percentile[1])

    hdr.update(dat_hdr)
    return hdr

def make_emitters(header_format: dict):
    PAD = {**{
        "REC_TEMP_LPAD": 6, "ACQ_LPAD": 7,
        "PIX_LPAD": 7, "PIX_SEP": 7,
        "E3_LPAD": 13, "E3_SEP": 13,
        "E6_LPAD": 11, "E6_SEP": 11,
        "OFF_LPAD": 13, "OFF_SEP": 9,
        "ANGLE_LPAD": 12, "BIAS_LPAD": 12,
        "TYPE1": 14, "TYPE2": 12,
        "DATE_LPAD": 1,
    }, **header_format.get("left_pads", {})}

    EOL = header_format.get('line_ending', '\n')
    if EOL in ('\\r\\n', r'\r\n'): EOL = '\r\n'
    if EOL in ('\\n', r'\n'):      EOL = '\n'

    def emit_SCANIT_TYPE(val: str) -> str:
        toks = (val or '').split()
        c1 = toks[0] if toks else 'FLOAT'
        c2 = toks[1] if len(toks) > 1 else 'LSBFIRST'
        return ' ' * PAD['TYPE1'] + c1 + ' ' * PAD['TYPE2'] + c2

    def emit_REC_DATE(val: str) -> str:
        return ' ' * PAD['DATE_LPAD'] + str(val).strip()

    def emit_REC_TIME(val: str) -> str:
        s = str(val).strip()
        parts = re.findall(r'\d{1,2}', s)
        return f"{int(parts[0]):02d}:{int(parts[1]):02d}:{int(parts[2]):02d}" if len(parts) >= 3 else s

    def emit_REC_TEMP(val: str) -> str:
        xs = _nums(val, 1)
        return ' ' * PAD['REC_TEMP_LPAD'] + f"{(xs[0] if xs else 0.0):.10f}"

    def emit_ACQ_TIME(val: str) -> str:
        xs = _nums(val, 1)
        return ' ' * PAD['ACQ_LPAD'] + f"{(xs[0] if xs else 0.0):.1f}"

    def emit_SCAN_PIXELS(val: str) -> str:
        Nx, Ny = map(int, _nums(val, 2))
        return ' ' * PAD['PIX_LPAD'] + f"{Nx}" + ' ' * PAD['PIX_SEP'] + f"{Ny}"

    def emit_SCAN_TIME(val: str) -> str:
        a, b = (_nums(val, 2) + [0.0, 0.0])[:2]
        return ' ' * PAD['E3_LPAD'] + _E(a, 3) + ' ' * PAD['E3_SEP'] + _E(b, 3)

    def emit_SCAN_RANGE(val: str) -> str:
        a, b = (_nums(val, 2) + [0.0, 0.0])[:2]
        return ' ' * PAD['E6_LPAD'] + _E(a, 6) + ' ' * PAD['E6_SEP'] + _E(b, 6)

    def emit_SCAN_OFFSET(val: str) -> str:
        a, b = (_nums(val, 2) + [0.0, 0.0])[:2]
        return ' ' * PAD['OFF_LPAD'] + _E(a, 6) + ' ' * PAD['OFF_SEP'] + _E(b, 6)

    def emit_SCAN_ANGLE(val: str) -> str:
        xs = _nums(val, 1)
        return ' ' * PAD['ANGLE_LPAD'] + _E(xs[0] if xs else 0.0, 3)

    def emit_BIAS(val: str) -> str:
        xs = _nums(val, 1)
        return ' ' * PAD['BIAS_LPAD'] + _E(xs[0] if xs else 0.0, 3)

    def emit_Z_CONTROLLER(val: str) -> str:
        lines = (val or "").splitlines()
        if lines and not lines[0].startswith('\t'):
            lines[0] = '\t' + lines[0]
        return EOL.join(ln.rstrip(' ') for ln in lines)

    def emit_DATA_INFO(val: str) -> str:
        lines = (val or "").splitlines()
        fixed = []
        for ln in lines:
            if ln and not ln.startswith("\t"):
                ln = "\t" + ln
            fixed.append(ln.rstrip(" "))
        return EOL.join(fixed) + EOL

    special = {
        'SCANIT_TYPE':  emit_SCANIT_TYPE,
        'REC_DATE':     emit_REC_DATE,
        'REC_TIME':     emit_REC_TIME,
        'REC_TEMP':     emit_REC_TEMP,
        'ACQ_TIME':     emit_ACQ_TIME,
        'SCAN_PIXELS':  emit_SCAN_PIXELS,
        'SCAN_TIME':    emit_SCAN_TIME,
        'SCAN_RANGE':   emit_SCAN_RANGE,
        'SCAN_OFFSET':  emit_SCAN_OFFSET,
        'SCAN_ANGLE':   emit_SCAN_ANGLE,
        'BIAS':         emit_BIAS,
        'Z-CONTROLLER': emit_Z_CONTROLLER,
        'DATA_INFO':    emit_DATA_INFO,
    }
    return special, EOL

# ═════════════════════════════════════════════════════════════════════════════
# ════════════ I M A G E  P R O C E S S I N G ═════════════════════════════════
# ═════════════════════════════════════════════════════════════════════════════

def image_size(hdr: Dict[str, str], dat: Optional[Path] = None) -> Tuple[float, float, float, float]:
    try:
        if 'Length x[A]' in hdr and 'Length y[A]' in hdr:
            width_nm = float(hdr['Length x[A]']) / 10.0
            height_nm = float(hdr['Length y[A]']) / 10.0
            Nx = int(find_hdr(hdr, 'Num.X', default_scaling_parameters.get('Num.X', 256)))
            Ny = int(find_hdr(hdr, 'Num.Y', default_scaling_parameters.get('Num.Y', 256)))
            dx_nm = width_nm / (Nx - 1)
            dy_nm = height_nm / (Ny - 1)
            return width_nm, height_nm, dx_nm, dy_nm

        Gx = float(find_hdr(hdr, 'GainX', default_scaling_parameters.get('GainX', 1)))
        Gy = float(find_hdr(hdr, 'GainY', default_scaling_parameters.get('GainY', 1)))
        Xp = float(find_hdr(hdr, 'XPiezoconst', default_scaling_parameters.get('XPiezoconst', 1)))
        Yp = float(find_hdr(hdr, 'YPiezoconst', default_scaling_parameters.get('YPiezoconst', 1)))

        Nx = int(find_hdr(hdr, 'Num.X', default_scaling_parameters.get('Num.X', 256)))
        Ny = int(find_hdr(hdr, 'Num.Y', default_scaling_parameters.get('Num.Y', 256)))

        V_per_DAC = 10.0 / 524288
        width_nm = V_per_DAC * Gx * Xp * Nx
        height_nm = V_per_DAC * Gy * Yp * Ny
        dx_nm = width_nm / (Nx - 1)
        dy_nm = height_nm / (Ny - 1)
        return width_nm, height_nm, dx_nm, dy_nm
    except Exception as e:
        print(f"Image size error: {e}")
        return 0.0, 0.0, 0.0, 0.0

def scale(data: np.ndarray, idx: int, hdr: Dict[str, str], dat: Optional[Path] = None) -> np.ndarray:
    Gz = float(find_hdr(hdr, 'GainZ', default_scaling_parameters['GainZ']))
    Zp = float(find_hdr(hdr, 'ZPiezoconst', default_scaling_parameters['ZPiezoconst']))
    Ps = 10 ** (float(find_hdr(hdr, 'GainPre', default_scaling_parameters['PreAmpSensi'])))

    if idx % 2 == 0:  # topography
        Dz = float(find_hdr(hdr, 'Dacto[A]z', data_dac_conversion_factor * Gz * Zp))
        data_nm = data * Dz
        data_m = data_nm * 1e-9
        return data_m
    else:  # current
        return (data * data_dac_conversion_factor / Ps)

def scale_all_chan(stack: np.ndarray, hdr: Dict[str, str], dat: Path, numChan: int) -> np.ndarray:
    scaled = np.zeros_like(stack)
    for i in range(numChan):
        scaled[i] = scale(stack[i], i, hdr, dat)
    return scaled

def process_dat(dat: Path) -> Tuple[Dict[str, str], List[Tuple[str, str, str, np.ndarray]], int]:
    raw = dat.read_bytes()
    dat_hdr_bytes, comp = raw.split(b'DATA', 1)

    dat_hdr = parse_header(dat_hdr_bytes)
    _ = image_size(dat_hdr, dat)

    Nx = int(find_hdr(dat_hdr, 'Num.X', default_scaling_parameters['Num.X']))
    Ny = int(find_hdr(dat_hdr, 'Num.Y', default_scaling_parameters['Num.Y']))

    payload = zlib.decompress(comp)
    try:
        arr = np.frombuffer(payload, dtype='<f4')[:4 * Ny * Nx]
        stack = arr.reshape((4, Ny, Nx))
        numChan = 4
    except ValueError:
        arr = np.frombuffer(payload, dtype='<f4')[:2 * Ny * Nx]
        stack = arr.reshape((2, Ny, Nx))
        numChan = 2

    trimmed_stack_raw, new_num_rows = trim(stack)
    dat_hdr['Num.Y'] = new_num_rows

    stack_post = equalise(scale_all_chan(trimmed_stack_raw, dat_hdr, dat, numChan))

    for i in range(numChan):
        img = stack_post[i]
        minp_val = np.percentile(img, percentile[0])
        maxp_val = np.percentile(img, percentile[1])
        stack_post[i] = np.clip(img, minp_val, maxp_val)

    FT = stack_post[0]
    FC = stack_post[1]
    if numChan == 2:
        BT = np.fliplr(stack_post[0])
        BC = np.fliplr(stack_post[1])
    else:
        BT = np.fliplr(stack_post[2])
        BC = np.fliplr(stack_post[3])

    hdr = construct_hdr(dat_hdr, dat, numChan)

    Ny2, Nx2 = FT.shape
    hdr["SCAN_PIXELS"] = f"{Nx2}{' '*7}{Ny2}"

    imgs = [
        ("Z", "m", "forward",  to_f32(FT)),
        ("Z", "m", "backward", to_f32(BT)),
        ("Current", "A", "forward",  to_f32(FC)),
        ("Current", "A", "backward", to_f32(BC)),
    ]
    return hdr, imgs, numChan

# ═════════════════════════════════════════════════════════════════════════════
# ════════════ S C R I P T | E X E C U T I V E ═════════════════════════════════
# ═════════════════════════════════════════════════════════════════════════════

def reconstruct_from_hdr_imgs(
    hdr: dict,
    imgs: list[tuple[str, str, str, np.ndarray]] | list[np.ndarray],
    header_format: dict,
    post_end_bytes: bytes,
    pre_payload_bytes: bytes,
    out_path: Path,
    tail_bytes: bytes = b"",
    force_data_offset: int | None = None,
    filler_char: bytes = b" "
):
    special, EOL = make_emitters(header_format)

    block_order = header_format.get('block_order')
    if not block_order:
        keys = [k for k in hdr.keys() if not k.startswith('__') and k != 'DATA_INFO_PARSED']
        di = [k for k in keys if k == 'DATA_INFO']
        rest = [k for k in keys if k != 'DATA_INFO']
        block_order = rest + di

    out_lines, emitted = [], set()

    def _emit_key(key: str):
        val = hdr[key]
        body = special[key](val) if key in special else str(val)
        if body and not body.endswith(EOL):
            body = body.rstrip(' ')
        out_lines.append(f":{key}:{EOL}{body}{EOL}")

    for key in block_order:
        if key not in hdr or key == 'DATA_INFO_PARSED' or re.fullmatch(r'\d+', key):
            continue
        _emit_key(key)
        emitted.add(key)

    for key in hdr:
        if key in emitted or key.startswith('__') or key == 'DATA_INFO_PARSED' or re.fullmatch(r'\d+', key):
            continue
        _emit_key(key)

    header_core = "".join(out_lines)
    while not header_core.endswith(EOL * 2):
        header_core += EOL

    hdr_bytes = header_core.encode('latin-1', 'ignore')

    if force_data_offset is not None:
        target_len = int(force_data_offset) - (len(b":SCANIT_END:") + len(post_end_bytes) + len(pre_payload_bytes))
        fill = target_len - len(hdr_bytes)
        if fill:
            trailer = (EOL * 2).encode('latin1')
            body_wo_final_blank = hdr_bytes[:-len(trailer)]
            one_eol = EOL.encode('latin1')
            if isinstance(filler_char, str):
                filler_char = filler_char.encode('latin1', 'ignore')
            hdr_bytes = body_wo_final_blank + one_eol + (filler_char * fill) + one_eol + one_eol

    header_bytes = hdr_bytes + b":SCANIT_END:" + post_end_bytes + pre_payload_bytes
    data_offset = len(header_bytes)

    toks = (hdr.get("SCANIT_TYPE") or "").split()
    endian = '>' if (len(toks) >= 2 and toks[1].strip().upper() == 'MSBFIRST') else '<'
    dt = np.dtype(endian + 'f4')
    Nx, Ny = map(int, _nums(hdr.get("SCAN_PIXELS", ""), 2))

    arrs = []
    for item in imgs:
        if isinstance(item, tuple):
            arrs.append(item[3])
        else:
            arrs.append(item)

    for i, a in enumerate(arrs):
        if a.shape != (Ny, Nx):
            raise ValueError(f"Plane {i} shape {a.shape} != {(Ny, Nx)}")

    payload = b"".join(np.asarray(a, dtype=dt, order='C').tobytes(order='C') for a in arrs)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(header_bytes + payload + (tail_bytes or b""))
    return data_offset, len(payload)

def executive_dat_to_sxm(dat: Path) -> None:
    hdr, imgs, numChan = process_dat(dat)
    layout, header_format = load()

    name_of_built_file = dat.stem + ".sxm"
    out_path = out_dir / name_of_built_file

    imgs_arrays = [arr for (_name, _unit, _dir, arr) in imgs]
    data_offset, payload_len = reconstruct_from_hdr_imgs(
        hdr=hdr,
        imgs=imgs_arrays,
        header_format=header_format,
        post_end_bytes=layout["post_end_bytes"],
        pre_payload_bytes=layout["pre_payload_bytes"],
        out_path=out_path,
        tail_bytes=layout["tail_bytes"],
        force_data_offset=layout["data_offset"],
        filler_char=b" "
    )
    print("data_offset:", data_offset, "payload_len:", payload_len)
    if numChan == 2:
        print(f"WARNING FOR {dat.stem}: Original .dat had 2 channels; duplicated BACKWARD planes were written to keep the .sxm output four-plane.")

def main() -> None:
    global dat_source, rebuilt_path, cushion_path, out_dir

    args = parse_args()
    dat_source = Path(args.input_dir or args.legacy_input or DEFAULT_INPUT_DIR)
    rebuilt_path = Path(args.output_dir or args.legacy_output or DEFAULT_OUTPUT_DIR)
    out_dir = rebuilt_path
    cushion_path = Path(args.cushion_dir or args.legacy_cushion or DEFAULT_CUSHION_DIR)

    apply_tuners()

    if dat_source.is_dir():
        error_log = []
        for dat_path in dat_source.glob("*.dat"):
            print("Processing file:", dat_path.name)
            try:
                executive_dat_to_sxm(dat_path)
            except Exception as e:
                error_log.append((dat_path.name, str(e)))
                print(f"Error processing {dat_path.name}: {e}")

        if error_log:
            print("\nErrors encountered during processing:")
            for fname, err in error_log:
                print(f"{fname}: {err}")
        else:
            print("All files processed successfully.")
    else:
        if dat_source.suffix.lower() == ".dat":
            print("Processing file:", dat_source.name)
            executive_dat_to_sxm(dat_source)
            print("File processed successfully.")
        else:
            raise ValueError(f"Must be a .dat file or a directory, got: {dat_source}")

if __name__ == "__main__":
    main()
