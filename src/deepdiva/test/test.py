from deepdiva.utils.h2p_utils import H2P

def main():
    
    # Testing h2p_translator
    h2p_translator = H2P()

    # test case 1 #####################################
    # when the preset has a varied number of parameters

    # Test parameters
    real_patch = [(0, 0.37), (1, 0.0), (2, 1.0), (3, 0.08), (4, 0.5714285714285714), (5, 0.16666666666666666), (6, 0.0), (7, 0.0), (8, 0.0), (9, 0.5), (10, 1.0), (11, 0.07692307692307693), (12, 0.07692307692307693), (13, 0.0), (14, 0.5), (15, 0.5), (16, 0.0), (17, 0.0), (18, 0.0), (19, 0.0), (20, 0.15), (21, 0.19), (22, 0.0), (23, 0.26), (24, 0.64), (25, 0.0), (26, 1.0), (27, 0.0), (28, 1.0), (29, 0.0), (30, 1.0), (31, 0.0), (32, 1.0), (33, 0.06), (34, 0.54), (35, 0.0), (36, 0.4), (37, 0.0), (38, 0.5), (39, 0.0), (40, 0.0), (41, 1.0), (42, 0.0), (43, 0.0), (44, 0.4), (45, 0.56), (46, 0.0), (47, 0.5), (48, 0.0), (49, 0.0), (50, 0.0), (51, 0.0), (52, 0.0), (53, 1.0), (54, 0.09), (55, 0.038461538461538464), (56, 1.0), (57, 0.14285714285714285), (58, 0.0), (59, 0.0), (60, 0.0), (61, 0.0), (62, 0.74), (63, 0.0), (64, 0.5), (65, 0.038461538461538464), (66, 0.3333333333333333), (67, 0.14285714285714285), (68, 0.0), (69, 0.0), (70, 0.0), (71, 0.0), (72, 0.62), (73, 0.0), (74, 0.5), (75, 0.0), (76, 0.0), (77, 0.0), (78, 0.0), (79, 0.0), (80, 0.0), (81, 0.0), (82, 0.0), (83, 0.0), (84, 0.0), (85, 0.0), (86, 0.5), (87, 0.6003333333333333), (88, 0.4995), (89, 0.0), (90, 0.5), (91, 0.5), (92, 0.5), (93, 0.8375), (94, 0.0), (95, 0.0), (96, 1.0), (97, 1.0), (98, 0.9), (99, 0.49), (100, 0.3333333333333333), (101, 0.0), (102, 0.0), (103, 0.5652173913043478), (104, 0.625), (105, 0.7391304347826086), (106, 0.5), (107, 0.0), (108, 0.5), (109, 0.7391304347826086), (110, 0.53), (111, 0.0), (112, 0.0), (113, 1.0), (114, 0.0), (115, 0.0), (116, 1.0), (117, 0.0), (118, 1.0), (119, 0.0), (120, 0.5), (121, 0.0), (122, 0.0), (123, 0.0), (124, 0.0), (125, 1.0), (126, 0.0), (127, 0.0), (128, 0.0), (129, 1.0), (130, 0.0), (131, 0.0), (132, 0.0), (133, 0.0), (134, 0.0), (135, 0.0), (136, 0.5), (137, 0.0), (138, 0.5), (139, 0.0), (140, 0.0), (141, 0.0), (142, 0.0), (143, 0.0), (144, 0.0), (145, 0.5), (146, 0.25), (147, 0.0), (148, 0.98), (149, 0.22), (150, 0.6521739130434783), (151, 0.4), (152, 0.043478260869565216), (153, 0.35), (154, 0.36), (155, 0.5), (156, 0.0), (157, 1.0), (158, 0.0), (159, 0.0), (160, 0.16), (161, 0.043478260869565216), (162, 0.62), (163, 0.0), (164, 0.5), (165, 0.0), (166, 0.5), (167, 0.5), (168, 0.2), (169, 1.0), (170, 0.6086956521739131), (171, 1.0), (172, 0.4782608695652174), (173, 0.66), (174, 0.0), (175, 0.0), (176, 0.5), (177, 0.5), (178, 0.0), (179, 1.0), (180, 0.5), (181, 0.5), (182, 0.79), (183, 0.0), (184, 0.5), (185, 0.0), (186, 0.5), (187, 0.0), (188, 0.0), (189, 1.0), (190, 0.0), (191, 1.0), (192, 0.8), (193, 0.5), (194, 0.7487437185929648), (195, 0.9), (196, 0.4), (197, 0.06666666666666667), (198, 0.2), (199, 0.2), (200, 0.2), (201, 0.0), (202, 0.25), (203, 0.0), (204, 1.0), (205, 1.0), (206, 0.5), (207, 0.0), (208, 1.0), (209, 0.5), (210, 0.0), (211, 1.0), (212, 0.5), (213, 0.3), (214, 0.85), (215, 0.5), (216, 0.0), (217, 0.5), (218, 0.0), (219, 0.0), (220, 0.5), (221, 1.0), (222, 0.0), (223, 0.5), (224, 0.0), (225, 0.5), (226, 0.0), (227, 0.0), (228, 1.0), (229, 0.144), (230, 1.0), (231, 0.77), (232, 0.23), (233, 0.5276381909547738), (234, 1.0), (235, 0.28), (236, 0.06666666666666667), (237, 0.2), (238, 0.2), (239, 0.2), (240, 0.0), (241, 0.25), (242, 0.0), (243, 1.0), (244, 1.0), (245, 0.5), (246, 0.0), (247, 1.0), (248, 0.5), (249, 0.0), (250, 1.0), (251, 0.5), (252, 0.3), (253, 0.85), (254, 0.5), (255, 0.0), (256, 0.3333333333333333), (257, 0.6666666666666666), (258, 0.0), (259, 0.2), (260, 0.3333333333333333), (261, 0.0), (262, 0.0), (263, 0.0), (264, 0.0), (265, 0.5), (266, 0.0), (267, 0.0), (268, 0.7391304347826086), (269, 0.5), (270, 0.0), (271, 0.0), (272, 0.0), (273, 0.0), (274, 1.0), (275, 0.5), (276, 1.0), (277, 0.5), (278, 0.0), (279, 0.0), (280, 0.0)]

    filename = "HS-Brighton.h2p"
    filename_path = f"../deepdiva/data/{filename}"
    patch = h2p_translator.preset_to_patch(filename_path, normalize=True)

    if patch == real_patch:
        print("Test 1 = Pass")
    else:
        print("Test 1 = Fail")
    ####################################################


if __name__ == "__main__":
    main()

