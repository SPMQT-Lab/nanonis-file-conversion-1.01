import argparse
import zlib, re, math, json, os
import numpy as np
from pathlib import Path
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "sample_input"

DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "output_png"

DEFAULT_ENDIAN = "MSBFIRST"


import shutil
# ————————————————————————————————————————


# helpers 

def _f(x, default=None):
    try: return float(str(x).replace(",", "."))
    except: return default

def _i(x, default=None):
    try: return int(float(str(x).replace(",", ".")))
    except: return default

def parse_header(hb: bytes) -> dict:
    hdr = {}
    for line in hb.splitlines():
        if b'=' in line:
            k, v = line.split(b'=', 1)
            key = k.decode('ascii', 'ignore').split('/')[-1].strip()
            val = v.decode('ascii', 'ignore').strip()
            hdr[key] = val
    return hdr

def find_hdr(hdr: dict, hint: str, default=None):
    for k in hdr:
        if hint.lower() in k.lower():
            return hdr[k]
    return default

def scan_dir(hdr: dict) -> str:
    v = str(find_hdr(hdr, "ScanYDirec", "1")).strip()
    return "down" if v == "1" else "up"

def sanitize(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', name).strip('_')

def get_dac_bits(H, default=20):
    """
    extract DAC resolution in bits from header.
    Accepts values like: '20bit', '20 bit', '20', 20
    Falls back to default if missing or unparseable.
    """
    raw = find_hdr(H, "DAC-Type", None)
    if raw is None:
        return default

    s = str(raw).lower().strip()

    # extract first integer found
    m = re.search(r"\d+", s)
    if m:
        try:
            return int(m.group())
        except ValueError:
            pass

    return default


# ---------- core builder ----------
def dat_to_hdr_imgs(dat_path: Path, out_dir: Path, endian="MSBFIRST"):

    raw = dat_path.read_bytes()
    hb, comp = raw.split(b"DATA", 1)
    H = parse_header(hb)
    
    # Nx, Ny
    Nx = _i(find_hdr(H, "Num.X", 0), 0)
    Ny = _i(find_hdr(H, "Num.Y", 0), 0)

    print('Nx, Ny:', Nx, Ny)
    #numChan = _i(find_hdr(H, "Channels", 4), 4)
    DAC_BITS = get_dac_bits(H, default=20)
    V_PER_DAC = 10.0 / (2**DAC_BITS)  # 10 V ref, 20-bit

    scanmode_sine = _i(find_hdr(H, "ScanmodeSine", 0), 0)
    if scanmode_sine != 0:
        raise NotImplementedError(
            "Sine scan mode not supported yet"
        )

    
    #print("Nx,Ny:", Nx, Ny)
    # decompress payload -> (4, Ny, Nx) as FT, FC, BT, BC (little-endian f32 typical)
    payload = zlib.decompress(comp)
    decompressed_bytes = len(payload)
    #expected_bytes = 4 * Ny * Nx * numChan
    #print('decompressed_bytes:', decompressed_bytes)
    #print('expected_bytes:', expected_bytes)
    #print('numChan:', numChan)
    #if decompressed_bytes != expected_bytes:
    #    raise ValueError(
    #        f"Decompressed data size mismatch: got {decompressed_bytes} bytes, "
    #        f"expected {expected_bytes} bytes (Nx={Nx}, Ny={Ny}, numChan={numChan})"
    #    )

    
    try:
        arr = np.frombuffer(payload, dtype="<f4", count= 4 * Ny * Nx).copy()  # make writable
        numChan = 4
    except:
        try:
            arr = np.frombuffer(payload, dtype="<f4", count= 2 * Ny * Nx).copy()  # make writable
            numChan = 2
        except Exception as e:
            print(f"Error reading payload: {e}")
            raise
    print('Number of channels:', numChan)

    stack = arr.reshape((numChan, Ny, Nx))
    #arr = np.concatenate([arr, arr], axis=0)   # this is to mimic 4 channels 

    #scan_x_mode = _i(find_hdr(H, "ScanXMode", 1), 1)

    #if scan_x_mode == 1: 
    #    #fast_axis = "x" 
    #    stack = arr.reshape((numChan, Ny, Nx))
    #else:
    #    # fast_axis = "y"
    #    stack = arr.reshape((numChan, Ny, Nx)).transpose((0, 2, 1)).reshape((-1,))
    #    Nx, Ny = Ny, Nx
    
    origin_upper = (str(find_hdr(H, "ScanYDirec", "1")).strip() == "1")

    
    



    # trim trailing invalid rows using FT (index 0)
    ch0 = stack[0]
    print(ch0.shape)
    valid_mask = np.logical_and(~np.isnan(ch0), ch0 != 0)
    rows, cols = np.where(valid_mask.reshape(Ny, Nx))
    if rows.size:
        last_row = int(rows.max())
        new_Ny = last_row + 1 if cols[rows == last_row].max() == (Nx - 1) else last_row
    else:
        new_Ny = Ny
    stack = stack[:, :new_Ny, :]
    Ny = new_Ny  # update header pixels

    # physical ranges
    Lx_A = _f(find_hdr(H, "Length x[A]", 0.0), 0.0)
    Ly_A = _f(find_hdr(H, "Length y[A]", 0.0), 0.0)
    Lx_m, Ly_m = Lx_A * 1e-10, Ly_A * 1e-10


    # ---- SCALE PAYLOAD TO SI, SET CAL=1 ----
    NEGATIVE_CURRENT = True  # set False if your sign convention is opposite

    # Z scale: m/DAC (prefer explicit Dacto[A]z)
    Dz_A_per_DAC = _f(find_hdr(H, "Dacto[A]z", None), None)
    if Dz_A_per_DAC is None:
        # Fallback if Dacto[A]z missing: use GainZ & ZPiezoconst (adjust if Zp not Å/V)
        Gz = _f(find_hdr(H, "GainZ", 10.0), 10.0)
        Zp = _f(find_hdr(H, "ZPiezoconst", 19.2), 19.2)  # Å/V in many setups
        Dz_A_per_DAC = V_PER_DAC * Gz * Zp * 1e2  # Å/DAC; if Zp is nm/V use 1e1 instead
    z_scale_m_per_DAC = Dz_A_per_DAC * 1e-10  # -> m/DAC

    # Current scale: A/DAC = V_PER_DAC / Preamp(V/A)
    gain_pow = _f(find_hdr(H, "GainPre",
                           _f(find_hdr(H, "GainPre 10^", 9), 9)), 9)
    preamp = 10.0 ** gain_pow  # V/A
    i_scale_A_per_DAC = ((-1.0) if NEGATIVE_CURRENT else 1.0) * (V_PER_DAC / preamp)

    # Apply scaling to the four channels (FT, FC, BT, BC) <- is the order
    for k in range(numChan):
        if k % 2 == 0:
            # Z channel
            stack[k] = (stack[k] * z_scale_m_per_DAC).astype(np.float32, copy=False)
        else:
            # I channel
            stack[k] = (stack[k] * i_scale_A_per_DAC).astype(np.float32, copy=False)
    
    # ---- imgs in Nanonis plane order: Zf, Zb, If, Ib ----
    if numChan == 4:
        z_fwd, i_fwd, z_bwd, i_bwd = stack[0], stack[1], stack[2], stack[3]
        imgs = [
            ("Z", "m", "forward",  z_fwd),
            ("Z", "m", "backward", z_bwd),
            ("Current", "A", "forward",  i_fwd),
            ("Current", "A", "backward", i_bwd),
        ]
    elif numChan == 2: 
        z_fwd, i_fwd = stack[0], stack[1]
        imgs = [
            ("Z", "m", "forward",  z_fwd),
            ("Current", "A", "forward",  i_fwd),
        ]

    # ---- persist hdr + PNGs ----
    
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "hdr.txt", "w", encoding="utf-8") as out:
        for line in hb.splitlines():
            if b"=" in line:
                key, val = line.split(b"=", 1)
                field = key.decode("ascii", "ignore").split("/")[-1].strip()
                out.write(f"{field}: {val.decode('ascii', 'ignore').strip()}\n")
    
    png_dir = out_dir / "pngs"
    png_dir.mkdir(parents=True, exist_ok=True)

    
    
    for k, (nm, un, dr, arr_) in enumerate(imgs):
        # display-time mirroring for backward scans
        disp = np.fliplr(arr_) if dr == "backward" else arr_

        # percentile clip on displayed array, finite only
        finite = np.isfinite(disp)
        if np.any(finite):
            vmin, vmax = np.percentile(disp[finite], [1, 99])
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin = float(np.nanmin(disp[finite]))
                vmax = float(np.nanmax(disp[finite]))
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                    vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = 0.0, 1.0

        # map to 8-bit
        x = np.array(disp, dtype=np.float32, copy=False)
        x = np.clip(x, vmin, vmax)
        if vmax > vmin:
            x = (x - vmin) / (vmax - vmin)
        else:
            x = x * 0.0
        u8 = (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

        # match matplotlib origin
        if origin_upper:
            u8 = np.flipud(u8)

        im = Image.fromarray(u8, mode="L")
        fname = f"img_{k:02d}_{sanitize(nm)}_{dr}.png"
        im.save(png_dir / fname)


    return {
        "Nx": Nx, "Ny": Ny,
        "Lx_m": Lx_m, "Ly_m": Ly_m,
        "z_scale_m_per_DAC": z_scale_m_per_DAC,
        "i_scale_A_per_DAC": i_scale_A_per_DAC,
        "out_dir": str(out_dir),
        "png_dir": str(png_dir),
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Nanonis .dat files to preview PNGs")
    parser.add_argument("legacy_input", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("legacy_output", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument(
        "--input-dir",
        dest="input_dir",
        default=None,
        help="Path to a .dat file or a directory containing .dat files",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Directory where per-file PNG folders will be written",
    )
    return parser.parse_args()


def main(src=None, out_root=None):
    if src is None and out_root is None:
        args = parse_args()
        src = args.input_dir or args.legacy_input or DEFAULT_INPUT_DIR
        out_root = args.output_dir or args.legacy_output or DEFAULT_OUTPUT_DIR
    else:
        src = src or DEFAULT_INPUT_DIR
        out_root = out_root or DEFAULT_OUTPUT_DIR

    SRC = Path(src)
    OUT_ROOT = Path(out_root)
    ENDIAN = DEFAULT_ENDIAN

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    def _run_one(dat_path: Path):
        out_dir = OUT_ROOT / dat_path.stem
        #out_dir = OUT_ROOT
        rep = dat_to_hdr_imgs(dat_path, out_dir, endian=ENDIAN)
        print(f"[OK] {dat_path.name} -> {rep['Nx']}x{rep['Ny']}  "
              f"Zscale={rep['z_scale_m_per_DAC']:.3e} m/DAC  Iscale={rep['i_scale_A_per_DAC']:.3e} A/DAC")
    
    error_files = {}

    if SRC.is_file():
         _run_one(SRC)
    else:
        files = sorted(SRC.glob("*.dat"))
        if not files:
            print(f"No .dat files in {SRC}")
        for p in files:
            print(f"Processing {p.name}...")
            try:
               _run_one(p)
            except Exception as e:
                error_files[p.name] = str(e)
                
        if error_files:
            print("Some files failed:")
            print(json.dumps(error_files, indent=2))
            
    print(f"Done. Outputs are in: {OUT_ROOT}")


# ========= F5 RUN BLOCK =========



if __name__ == "__main__":
    main()

