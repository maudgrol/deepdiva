def scale_placeholder(x):
    return x

def scale_float_0_100(x):
    return x*100

def scale_float_0_150(x):
    return x*150

def scale_float_0_200(x):
    return x*200

def scale_float_24_24(x):
    return (x*48)-24

def scale_float_30_30(x):
    return (x*60)-30

def scale_float_120_120(x):
    return (x*240)-120

def scale_float_30_150(x):
    return (x*120)+30

def scale_int_0_1(x):
    res = int(2 * x)
    if res == 2:
        res = 1
    return res

def scale_int_0_2(x):
    res = int(3 * x)
    if res == 3:
        res = 2
    return res

def scale_int_0_3(x):
    res = int(4 * x)
    if res == 4:
        res = 3
    return res

def scale_int_0_4(x):
    res = int(5 * x)
    if res == 5:
        res = 4
    return res

def scale_int_0_22(x):
    res = int(23 * x)
    if res == 23:
        res = 22
    return res

def scale_int_1_2(x):
    res = int(2 * x)+1
    if res == 3:
        res = 2
    return res

def scale_int_1_4(x):
    res = int(4 * x)+1
    if res == 5:
        res = 4
    return res

def scale_fixed_octave(x):
    res = int(5 * x)
    if res == 5:
        res = 4
    return (res-2) * 12