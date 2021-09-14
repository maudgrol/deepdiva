from utils.transformer import preset_to_patch


x= preset_to_patch("../data/h2p_example.h2p")
print(x)

#if you like to NOT normalize the preset to patch
x= preset_to_patch("../data/h2p_example.h2p", normal=False)
print(x)