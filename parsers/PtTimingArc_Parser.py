def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
class timing_arc:
    def __init__(self):
        self.from_pin = ""
        self.to_pin = ""
        self.is_cell = True
        self.delay = [None, None, None, None]
        self.sense_unate = ""
        self.when = ""
        self.sdf_cond = ""
    def __repr__(self):
        timing_arc_repr = f"Max_Rise:{self.delay[0]}, Max_Fall:{self.delay[1]}, Min_Rise:{self.delay[2]}, Min_Fall:{self.delay[3]}"
        return f"timing_arc(from_pin={self.from_pin}, to_pin={self.to_pin}, is_cell={self.is_cell}, sense={self.sense_unate}, when={self.when}, sdf_cond={self.sdf_cond}, delay=[{timing_arc_repr}]"

def Read_PtTimingArc(inrpt):
    """Read timing arcs from PT report, return list of timing arcs with delay and sense unate on them"""
    timing_arcs = []
    flag_start = False
    with open(inrpt, 'r') as infile:
        for line in infile:
            index = line.split()
            if len(index) > 0:
                if index[0] == "Information:":
                    continue
                if flag_start and "---" not in index[0]:
                    new_timingarc = timing_arc()
                    new_timingarc.sense_unate = index[5]
                    new_timingarc.from_pin = index[6]
                    new_timingarc.to_pin = index[7]
                    new_timingarc.when = index[8]
                    new_timingarc.sdf_cond = index[9]
                    for i in range(4):
                        if is_float(index[i]):
                            new_timingarc.delay[i] = float(index[i])
                    if index[4] == "true":
                        new_timingarc.is_cell = True
                    else:
                        new_timingarc.is_cell = False
                    timing_arcs.append(new_timingarc)
                if index[0] == "Max_Rise":
                    flag_start = True
    
    return timing_arcs