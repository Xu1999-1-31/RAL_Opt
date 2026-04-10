def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
class cell_leakage:
    def __init__(self):
        self.name = ''
        self.leakage = [None, None, None] # Gate_Leakage, Intrinsic_Leakage,Totgal_Leakage
    def __repr__(self):
        leakage_repr = f"Gate_Leakage:{self.leakage[0]}, Intrinsic_Leakage:{self.leakage[1]}, Totgal_Leakage:{self.leakage[2]}"
        return f"cell_leakage(cell_name={self.name}, leakage=[{leakage_repr}]"

def Read_PtLeakage(inrpt):
    """Read cell leakage power from PT report, return list of cells with leakage power on them"""
    cells = {}
    flag_start = False
    with open(inrpt, 'r') as infile:
        for line in infile:
            index = line.split()
            if len(index) > 0:
                if index[0] == "Information:":
                    continue
                if flag_start and "---" not in index[0]:
                    newcell = cell_leakage()
                    newcell.name = index[3]
                    for i in range(3):
                        if is_float(index[i]):
                            newcell.leakage[i] = float(index[i])
                    cells[newcell.name] = newcell
                if index[0] == "Gate_Leakage":
                    flag_start = True
    return cells