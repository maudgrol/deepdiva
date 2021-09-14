from utils.transformer import preset_to_patch
from utils.transformer import patch_to_preset


x = preset_to_patch("/Users/grahamherdman/Documents/data-science-retreat/deep-diva/deepdiva/data/MS-Rev1.h2p")
# print(x)

# #if you like to NOT normalize the preset to patch
# x_norm = preset_to_patch("../data/h2p_example.h2p", normal=False)
# print(x_norm)


x_preset = patch_to_preset(x, "/Users/grahamherdman/Documents/data-science-retreat/deep-diva/deepdiva/data/patch_to_preset_test.h2p")
print(x_preset)
