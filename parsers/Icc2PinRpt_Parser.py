from dataclasses import dataclass, field
from typing import List, Optional
@dataclass
class Icc2_Pin():
    name: str
    llx: float
    lly: float
    urx: float
    ury: float

def Read_Icc2PinRpt(inrpt):
    """Read pin properties from Icc2 report, return dict of Icc2_pin with propertys"""
    Icc2_pins = {}
    flag_start = False
    with open(inrpt, 'r') as infile:
        for line in infile:
            index = line.split()
            if len(index) > 0:
                if flag_start and "---" not in index[0]:
                    newpin = Icc2_Pin(index[4], float(index[0]), float(index[1]),  float(index[2]),  float(index[3]))
                    Icc2_pins[newpin.name] = newpin
                if index[0] == "llx":
                    flag_start = True
    return Icc2_pins

if __name__ == "__main__":
    import sys 
    import os
    sys.path.append("../")
    # from Global_var import *
    import work.work_var
    file = os.path.join(work.work_var.icc2_data_dir, f"aes_cipher_top_1.0/rpt/aes_cipher_top_pin.rpt")
    pins = Read_Icc2PinRpt(file)
    print(pins)