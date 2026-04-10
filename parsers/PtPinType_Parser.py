def Read_PtPinType(inrpt):
    """Read pin type from PT report, return dict of pins with type list [is_async_pin is_clock_network]"""
    pins = {}
    flag_start = False
    with open(inrpt, 'r') as infile:
        for line in infile:
            index = line.split()
            if len(index) > 0:
                if index[0] == "Information:":
                    continue
                if flag_start and "---" not in index[0]:
                    if index[0] == "false":
                        type = [False]
                    else:
                        type = [True]
                    if index[1] == "false":
                        type.append(False)
                    else:
                        type.append(True)
                    pins[index[2]] = type
                if index[0] == "is_async_pin":
                    flag_start = True
    return pins