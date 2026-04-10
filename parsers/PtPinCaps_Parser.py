def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
class pin_ceff:
    def __init__(self):
        self.name = ''
        self.ceff = [None, None, None, None] # Max_Rise, Max_Fall, Min_Rise, Min_Fall
    def __repr__(self):
        ceff_repr = f"Max_Rise_Ceff:{self.ceff[0]}, Max_Fall_Ceff:{self.ceff[1]}, Min_Rise_Ceff:{self.ceff[2]}, Min_Fall_Ceff:{self.ceff[3]}"
        return f"Ceff(pin_name={self.name}, ceff=[{ceff_repr}]"

def Read_PtPinCaps(inrpt):
    """Read pin effective capacitance from PT report, return list of pins with ceff on them"""
    pins = {}
    flag_start = False
    with open(inrpt, 'r') as infile:
        for line in infile:
            index = line.split()
            if len(index) > 0:
                if index[0] == "Information:":
                    continue
                if flag_start and "---" not in index[0]:
                    newpin = pin_ceff()
                    newpin.name = index[4]
                    for i in range(4):
                        if is_float(index[i]):
                            newpin.ceff[i] = float(index[i])
                    pins[newpin.name] = newpin
                if index[0] == "cached_ceff_max_rise":
                    flag_start = True
    return pins