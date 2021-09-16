from utils.patch_utils import preset_to_patch
from utils.patch_utils import patch_to_preset

x = preset_to_patch("../data/MS-Rev1_deepdiva.h2p")
#x = preset_to_patch("../data/HS-Brighton.h2p")

# print(x)

# #if you like to NOT normalize the preset to patch
# x_norm = preset_to_patch("../data/h2p_example.h2p", normal=False)
# print(x_norm)


x_preset = patch_to_preset(x, "../data/patch_to_preset_test.h2p")
print(x_preset)
