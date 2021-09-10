def norm_placeholder(x):
    return 0

def norm_float_0_100(x):
    return x/100

def norm_float_0_150(x):
    return x/150

def norm_float_0_200(x):
    return x/200

def norm_float_24_24(x):
    return (x+24) / 48

def norm_float_30_30(x):
    return (x+30) / 60

def norm_float_120_120(x):
    return (x+120) / 240

def norm_float_30_150(x):
    return (x-30) / 120

def norm_int_0_1(x):
    return x

def norm_int_0_2(x):
    return x/2

def norm_int_0_3(x):
    return x/3

def norm_int_0_4(x):
    return x/4

def norm_int_0_22(x):
    return x/22

def norm_int_1_2(x):
    return x-1

def norm_int_1_4(x):
    return (x-1) /3

def norm_fixed_octave(x):
    return (x+24) / 48