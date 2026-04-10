def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
class pin_slack:
    def __init__(self):
        self.name = ''
        self.slack = [None, None, None, None] # Max_Rise, Max_Fall, Min_Rise, Min_Fall
    def __repr__(self):
        slack_repr = f"Max_Rise:{self.slack[0]}, Max_Fall:{self.slack[1]}, Min_Rise:{self.slack[2]}, Min_Fall:{self.slack[3]}"
        return f"pin_slack(pin_name={self.name}, slack=[{slack_repr}]"


def Read_PtPinSlack(inrpt):
    """Read pin slacks from PT report, return list of pins with slack on them and timing unit"""
    pins = {}
    flag_start = False
    with open(inrpt, 'r') as infile:
        for line in infile:
            index = line.split()
            if len(index) > 0:
                if index[0] == "Information:":
                    continue
                if flag_start and "---" not in index[0] and len(index) == 5:
                    newpin = pin_slack()
                    newpin.name = index[4]
                    for i in range(4):
                        if is_float(index[i]):
                            newpin.slack[i] = float(index[i])
                    pins[newpin.name] = newpin
                if index[0] == "Max_Rise":
                    flag_start = True
    return pins