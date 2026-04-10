def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
class port_trans:
    def __init__(self):
        self.name = ''
        self.trans = [None, None, None, None] # Max_Rise, Max_Fall, Min_Rise, Min_Fall
    def __repr__(self):
        trans_repr = f"Max_Rise:{self.trans[0]}, Max_Fall:{self.trans[1]}, Min_Rise:{self.trans[2]}, Min_Fall:{self.trans[3]}"
        return f"port_trans(port_name={self.name}, trans=[{trans_repr}]"

def Read_PtPortTrans(inrpt):
    """Read port transition time from PT report, return dict of ports with transition time on them"""
    ports = {}
    flag_start = False
    with open(inrpt, 'r') as infile:
        for line in infile:
            index = line.split()
            if len(index) > 0:
                if index[0] == "Information:":
                    continue
                if flag_start and "---" not in index[0]:
                    newport = port_trans()
                    newport.name = index[4]
                    for i in range(4):
                        if is_float(index[i]):
                            newport.trans[i] = float(index[i])
                    ports[newport.name] = newport
                if index[0] == "Max_Rise":
                    flag_start = True
    return ports