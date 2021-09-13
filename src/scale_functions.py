def scale_placeholder(x):
    return x

def scale_float_0_1(x):
    return x

def scale_float_0_4(x):
    return x*4

def scale_float_0_100(x):
    return x*100

def scale_float_0_150(x):
    return x*150

def scale_float_0_200(x):
    return x*200

def scale_float_1_9(x):
    return (x*8)+1

def scale_float_1_16(x):
    return (x*15)+1

def scale_float_1_200(x):
    return (x*199) + 1

def scale_float_2_100(x):
    return (x*98) +2

def scale_float_5_5(x):
    return (x*10) - 5

def scale_float_8_8(x):
    return (x*16) - 8

def scale_float_20_20(x):
    return (x*40) - 20

def scale_float_24_24(x):
    return (x*48) - 24

def scale_float_30_30(x):
    return (x*60) - 30

def scale_float_30_150(x):
    return (x*120) + 30

def scale_float_50_50(x):
    return (x*100) - 50

def scale_float_50_200(x):
    return (x*150) + 50

def scale_float_100_100(x):
    return (x*200) - 100

def scale_float_120_120(x):
    return (x*240) - 120

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

def scale_int_0_5(x):
    res = int(6 * x)
    if res == 6:
        res = 5
    return res

def scale_int_0_6(x):
    res = int(7 * x)
    if res == 7:
        res = 6
    return res

def scale_int_0_7(x):
    res = int(8 * x)
    if res == 8:
        res = 7
    return res

def scale_int_0_12(x):
    res = int(13 * x)
    if res == 13:
        res = 12
    return res

def scale_int_0_23(x):
    res = int(24 * x)
    if res == 24:
        res = 23
    return res

def scale_int_0_26(x):
    res = int(27 * x)
    if res == 27:
        res = 26
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

def scale_int_3_23(x):
    res = int(21 * x)+3
    if res == 24:
        res = 23
    return res

def scale_int_24_24(x):
    res = int(49 * x)-24
    if res == 25:
        res = 24
    return res

def scale_fixed_octave(x):
    res = int(5 * x)
    if res == 5:
        res = 4
    return (res-2) * 12

def scale_int_m1_3(x):
    res = int(5 * x) - 1
    if res == 4:
        res = 3
    return res

def scale_int_m3_23(x):
    res = int(27 * x) - 3
    if res == 24:
        res = 23
    return res












