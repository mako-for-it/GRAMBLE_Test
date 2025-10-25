def eighty_1(Code_ID, text):
    if Code_ID == "81A-1":
        code = "SOV"
    elif Code_ID == "81A-2":
        code = "SVO"
    elif Code_ID == "81A-3":
        code = "VSO"
    elif Code_ID == "81A-4":
        code = "VOS"
    elif Code_ID == "81A-5":
        code = "OVS"
    elif Code_ID == "81A-6":
        code = "OSV"
    elif Code_ID == "81A-7":
        code = "Lacking a dominant word order"
    elif Code_ID == "81B-1":
        code = "SOV or SVO"
    elif Code_ID == "81B-2":
        code = "VSO or VOS"
    elif Code_ID == "81B-3":
        code = "SVO or VSO"
    elif Code_ID == "81B-4":
        code = "SVO or VOS"
    elif Code_ID == "81B-5":
        code = "SOV or OVS"
    else:
        code = "skip"
    