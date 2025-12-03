#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, datetime as dt, os, re, signal, subprocess, sys, time
from statistics import mean

def which(cmd):
    from shutil import which as _which
    return _which(cmd)

def detect_xbutil():
    xb = which("xbutil") or which("xbutil2")
    if not xb:
        sys.exit("Error: xbutil / xbutil2 not found")
    return xb

def run_cmd(cmd, timeout=10):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=timeout, text=True)
        return out
    except subprocess.CalledProcessError as e:
        return e.output
    except subprocess.TimeoutExpired:
        return ""

def parse_power_from_electrical(text):
    """
    Parse the “electrical” report text, return:
      running_power (float or None),
      max_power (float or None),
      rails dict (rail → (voltage, current))
    """
    running_w = None
    max_w = None
    rails = {}

    # Match Max Power
    m_max = re.search(r"Max\s+Power\s*:\s*([\d\.]+)\s*Watts?", text, re.IGNORECASE)
    if m_max:
        max_w = float(m_max.group(1))

    # Match running Power (but not Max Power)
    # Use a pattern that ensures the key is exactly “Power” (not preceded by “Max”)
    # One approach: use a negative lookbehind to avoid “Max ”
    m_run = re.search(r"(?<!\w)(?<!Max\s)Power\s*:\s*([\d\.]+)\s*Watts?", text, re.IGNORECASE)
    if m_run:
        running_w = float(m_run.group(1))

    # Parse rails (Voltage / Current) lines
    voltage_re = re.compile(r"([\w\s\.0-9]+?)\s*:\s*([\d\.]+)\s*V", re.IGNORECASE)
    current_re = re.compile(r"([\w\s\.0-9]+?)\s*:\s*([\d\.]+)\s*A", re.IGNORECASE)

    # But note: lines like “12 Volts Auxillary : 12.197 V,  1.896 A”
    # So also a combined pattern:
    #   RailName : <voltage> V, <current> A
    combo_re = re.compile(r"([\w\s\.0-9]+?)\s*:\s*([\d\.]+)\s*V\s*,\s*([\d\.]+)\s*A", re.IGNORECASE)

    for line in text.splitlines():
        line = line.strip()
        # try combo first
        mc = combo_re.match(line)
        if mc:
            rail = mc.group(1).strip()
            volt = float(mc.group(2))
            curr = float(mc.group(3))
            rails[rail] = (volt, curr)
            continue
        # else separate parse
        mv = voltage_re.match(line)
        if mv:
            rail = mv.group(1).strip()
            volt = float(mv.group(2))
            v, c = rails.get(rail, (None, None))
            rails[rail] = (volt, c)
        mc2 = current_re.match(line)
        if mc2:
            rail = mc2.group(1).strip()
            curr = float(mc2.group(2))
            v, c = rails.get(rail, (None, None))
            rails[rail] = (v, curr)

    return running_w, max_w, rails

def build_cmd(xb, device):
    return [xb, "examine", "-d", device, "--report", "electrical"], "electrical"

def sample_once(xb, device):
    cmd, mode = build_cmd(xb, device)
    out = run_cmd(cmd)
    run_w, max_w, rails = parse_power_from_electrical(out)
    return out, run_w, max_w, rails, mode

def main():
    parser = argparse.ArgumentParser(description="U280 Power Logger (electrical)")
    parser.add_argument("--device", default="0")
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--duration", type=float, default=0)
    parser.add_argument("--out", default="u280_power_log.csv")
    parser.add_argument("--workload", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    xb = detect_xbutil()
    wl = None
    if args.workload and len(args.workload) > 0:
        print(f"[INFO] Launching workload: {' '.join(args.workload)}")
        wl = subprocess.Popen(args.workload, stdout=sys.stdout, stderr=sys.stderr, preexec_fn=os.setsid)

    f = open(args.out, "w", newline="")
    writer = csv.writer(f)
    # header to be extended later
    header = ["timestamp", "elapsed_s", "power_W", "max_power_W"]
    writer.writerow(header)
    f.flush()

    print(f"[INFO] Sampling every {args.interval:.3f} s → {args.out}")
    t0 = time.time()
    records = []
    rail_names = None
    stop = False
    def sig_handler(sig, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, sig_handler)

    try:
        while True:
            now = dt.datetime.now().isoformat(sep=" ", timespec="seconds")
            out, pw, mw, rails, mode = sample_once(xb, args.device)
            elapsed = time.time() - t0

            if rail_names is None:
                rail_names = sorted(rails.keys())
                # extend header
                for rn in rail_names:
                    header.append(f"{rn}_V")
                    header.append(f"{rn}_A")
                # rewrite header
                f.seek(0)
                writer = csv.writer(f)
                writer.writerow(header)
                f.flush()

            row = [now, f"{elapsed:.3f}", f"{pw:.6f}" if pw is not None else "", f"{mw:.6f}" if mw is not None else ""]
            for rn in rail_names:
                v, c = rails.get(rn, (None, None))
                row.append(f"{v:.6f}" if v is not None else "")
                row.append(f"{c:.6f}" if c is not None else "")

            writer.writerow(row)
            f.flush()

            if pw is not None:
                records.append(pw)

            if args.duration > 0 and elapsed >= args.duration:
                break
            if stop:
                break
            if wl and wl.poll() is not None:
                time.sleep(args.interval)
                _o, pw2, mw2, rails2, _m2 = sample_once(xb, args.device)
                now2 = dt.datetime.now().isoformat(sep=" ", timespec="seconds")
                elapsed2 = time.time() - t0
                row2 = [now2, f"{elapsed2:.3f}", f"{pw2:.6f}" if pw2 is not None else "", f"{mw2:.6f}" if mw2 is not None else ""]
                for rn in rail_names:
                    v2, c2 = rails2.get(rn, (None, None))
                    row2.append(f"{v2:.6f}" if v2 is not None else "")
                    row2.append(f"{c2:.6f}" if c2 is not None else "")
                writer.writerow(row2)
                f.flush()
                if pw2 is not None:
                    records.append(pw2)
                break

            time.sleep(max(0.0, args.interval))
    finally:
        if wl and wl.poll() is None:
            try:
                os.killpg(os.getpgid(wl.pid), signal.SIGTERM)
            except Exception:
                pass
        f.close()

    # statistics
    if records:
        print("\n=== Sampling Statistics ===")
        print(f"Samples: {len(records)}")
        print(f"Mean running power: {mean(records):.6f} W")
        print(f"Max running power: {max(records):.6f} W")
        print(f"Min running power: {min(records):.6f} W")
        print(f"CSV saved: {args.out}")
    else:
        print("No valid running Power readings found (could not parse “Power : xxx Watts”).")

if __name__ == "__main__":
    main()


# python u280_power_logger.py --device 0000:a1:00.1 --interval 0.1 --duration 30