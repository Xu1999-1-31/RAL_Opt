import re

def Read_GlobalRpt(inrpt):
    """Read global timing data from PT report report_global_timing, return wns tns"""
    wns = [0.0, 0.0]   # [setup, hold]
    tns = [0.0, 0.0]
    
    current_mode = None  # 0 setup, 1 hold

    pattern_wns = re.compile(r'^WNS\s+(-?\d+\.\d+)')
    pattern_tns = re.compile(r'^TNS\s+(-?\d+\.\d+)')
    
    with open(inrpt, 'r') as f:
        for line in f:
            line = line.strip()

            if line.startswith("Setup violations"):
                current_mode = 0
                continue

            if line.startswith("Hold violations"):
                current_mode = 1
                continue

            if current_mode is None:
                continue

            m = pattern_wns.match(line)
            if m:
                wns[current_mode] = float(m.group(1))
                continue

            m = pattern_tns.match(line)
            if m:
                tns[current_mode] = float(m.group(1))
                continue

    return wns, tns