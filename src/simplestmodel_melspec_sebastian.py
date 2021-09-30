
# items to be removed
unwanted_num = {0: "Overall Volume",
                3 : "LED Colour",
                16: "Note priority"
                256: "CLK",
                257: "CLK",
                258: "CLK",
                259: "ARP",
                260: "ARP",
                261: "ARP",
                262: "ARP",
                263: "ARP on/off",
                271: "ARP",
                43: "Key Follow" #makes no sense if no key change
                54: "Key Follow" #makes no sense if no key change
                }

a lot of parameters will start to make more sense if
(1) the recorded audio is longer than 1 second
(2) the difference between played and recorded audio is >0

if we put (1) on 3 seconds and (2) on 6 seconds we might have a really interesting dataset. LSTM would start to make more sense here though
prediction will become harder but the model....


all_vst_parameters = list(range(281))

trainable_vst_parameters = [p for p in all_vst_parameters if p not in unwanted_num.keys()]

