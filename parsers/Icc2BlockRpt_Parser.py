from dataclasses import dataclass, field
from typing import List, Optional
@dataclass
class Icc2_Block():
    name: str
    llx: float
    lly: float
    urx: float
    ury: float

def Read_Icc2BlockRpt(inrpt):
    """Read block properties from Icc2 report, return Icc2Block class with propertys"""
    flag_start = False
    with open(inrpt, 'r') as infile:
        for line in infile:
            index = line.split()
            if len(index) > 0:
                if flag_start and "---" not in index[0]:
                    block = Icc2_Block(index[4], float(index[0]), float(index[1]),  float(index[2]),  float(index[3]))
                if index[0] == "llx":
                    flag_start = True
    return block

if __name__ == "__main__":
    import sys 
    import os
    sys.path.append("../")
    # from Global_var import *
    import work.work_var
    file = os.path.join(work.work_var.icc2_data_dir, f"aes_cipher_top_1.0/rpt/aes_cipher_top_block.rpt")
    pins = Read_Icc2BlockRpt(file)
    print(pins)