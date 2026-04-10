def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
class pin_acutal_trans:
    def __init__(self):
        self.name = ''
        self.trans = [None, None, None, None] # Max_Rise, Max_Fall, Min_Rise, Min_Fall
    def __repr__(self):
        trans_repr = f"Max_Rise:{self.trans[0]}, Max_Fall:{self.trans[1]}, Min_Rise:{self.trans[2]}, Min_Fall:{self.trans[3]}"
        return f"pin_acutal_trans(pin_name={self.name}, trans=[{trans_repr}]"

def Read_PtPinTrans(inrpt):
    """Read pin actual trans time from PT report, return list of pins with actual tran on them"""
    pins = {}
    flag_start = False
    with open(inrpt, 'r') as infile:
        for line in infile:
            index = line.split()
            if len(index) > 0:
                if index[0] == "Information:":
                    continue
                if flag_start and "---" not in index[0]:
                    newpin = pin_acutal_trans()
                    newpin.name = index[4]
                    for i in range(4):
                        if is_float(index[i]):
                            newpin.trans[i] = float(index[i])
                    pins[newpin.name] = newpin
                if index[0] == "Max_Rise":
                    flag_start = True
    return pins