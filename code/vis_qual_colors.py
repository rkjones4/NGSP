import os, sys, data_utils, utils

red_l = (255, 152, 150)
red_d = (216, 37, 38)
red_o =(200, 82, 0)
cardinal = (177,3,24)
green_sl = (152, 223, 138)
green_l = (103, 191, 92)
green_d = (44, 160, 44)
yellow_d = (255, 193, 86)
yellow_l = (219, 219, 141)
blue_l = (158, 218, 229)
blue_n = (31, 119, 180)
blue_a = (39, 190, 207)
purple_d = (123, 102, 210)
purple_p = (148, 103, 189)
fush = (220, 95, 189)
orange_l = (255, 187, 120)
blue_la = (158, 218, 229)

dark_brown = (101, 67, 33)
pink = (255, 192, 203)
cafe = (161,130, 98)
mocha = (190, 164, 147)
teal = (0, 128, 128)
baby_blue = (137, 207, 240)
iceberg = (113, 166, 210)
turqoise = (64, 224, 208)
gold = (204, 164, 61)
lime = (206, 250, 5)
mint = (265, 255, 214)

DEFAULT = (155, 155, 155)

# FOR SEMANTIC

CMAP = {
    # VASE

    'pot/base/foot_base/foot': orange_l,
    'pot/body/container': blue_n,
    'pot/containing_things/liquid_or_soil': dark_brown,
    'pot/containing_things/plant': green_l,


    'cutting_instrument/knife/handle_side/guard': purple_d,
    'cutting_instrument/knife/handle_side/handle': blue_l,
    'cutting_instrument/knife/handle_side/butt': fush,
    
    'cutting_instrument/knife/blade_side/bolster': red_o,
    'cutting_instrument/knife/blade_side/blade': yellow_d,

    'cutting_instrument/dagger/blade_side/blade': orange_l,
    'cutting_instrument/dagger/handle_side/guard': blue_n,
    'cutting_instrument/dagger/handle_side/butt': purple_p ,
    'cutting_instrument/dagger/handle_side/handle': green_sl,

    # TABLE
    
    'table/regular_table/tabletop/tabletop_surface': red_d,
    'table/regular_table/tabletop/tabletop_dropleaf': red_o,
    'table/regular_table/tabletop/tabletop_frame': red_l,
    
    'table/regular_table/table_base/regular_leg_base/tabletop_connector': fush,
    'table/regular_table/table_base/regular_leg_base/leg': blue_n,
    'table/regular_table/table_base/regular_leg_base/foot': blue_a,
    'table/regular_table/table_base/regular_leg_base/runner': blue_l,
    'table/regular_table/table_base/regular_leg_base/bar_stretcher': teal,
    'table/regular_table/table_base/regular_leg_base/circular_stretcher': iceberg,
    'table/regular_table/table_base/pedestal_base/pedestal': blue_la,
    'table/regular_table/table_base/pedestal_base/tabletop_connector': baby_blue,
    
    'table/regular_table/table_base/drawer_base/bar_stretcher': green_l,
    'table/regular_table/table_base/drawer_base/shelf': green_d,
    'table/regular_table/table_base/drawer_base/vertical_side_panel': yellow_l,
    'table/regular_table/table_base/drawer_base/vertical_divider_panel': yellow_d,
    'table/regular_table/table_base/drawer_base/bottom_panel': orange_l,
    'table/regular_table/table_base/drawer_base/leg': turqoise,
    'table/regular_table/table_base/drawer_base/caster': lime,
    'table/regular_table/table_base/drawer_base/drawer': purple_d,
    'table/regular_table/table_base/drawer_base/back_panel': purple_p,
    'table/regular_table/table_base/drawer_base/cabinet_door': gold,
    'table/regular_table/table_base/drawer_base/vertical_front_panel': dark_brown,
    'table/regular_table/table_base/drawer_base/tabletop_connector': cardinal,
    'table/regular_table/table_base/drawer_base/foot': mint,
    
    'table/game_table/ping_pong_table/tabletop/tabletop_surface': pink,    
    'table/game_table/ping_pong_table/table_base/regular_leg_base/leg': cafe,        
    'table/game_table/ping_pong_table/ping_pong_net': mocha,
    
    # CHAIR
        
    'chair/chair_back/back_surface': red_l,
    'chair/chair_back/back_frame': red_d,
    'chair/chair_back/back_support':red_o,
    'chair/chair_back/back_connector': cardinal,
    
    'chair/chair_arm/arm_holistic_frame': green_sl,
    'chair/chair_arm/arm_horizontal_bar': green_l,
    'chair/chair_arm/arm_near_vertical_bar': green_d,
    'chair/chair_arm/arm_sofa_style': purple_p,

    'chair/chair_arm/arm_writing_table': fush,
    'chair/chair_arm/arm_connector': purple_d,
    
    
    'chair/chair_seat/seat_surface': yellow_d,
    'chair/chair_seat/seat_support': yellow_l,
    'chair/chair_seat/seat_frame': orange_l,
    
    'chair/chair_base/regular_leg_base/leg': blue_l,  
    'chair/chair_base/regular_leg_base/bar_stretcher': blue_n,
    'chair/chair_base/regular_leg_base/rocker': blue_a,
    'chair/chair_base/regular_leg_base/runner': blue_la,

    'chair/chair_base/foot_base/foot': pink,

    'chair/chair_base/star_leg_base/central_support': cafe,
    'chair/chair_base/star_leg_base/star_leg_set': mocha,
    
    
    
    # LAMP

    'lamp/ceiling_lamp/chandelier/lamp_unit_group/lamp_unit/lamp_head': red_l,
    'lamp/ceiling_lamp/chandelier/lamp_unit_group/lamp_unit/lamp_arm': red_d,
    'lamp/ceiling_lamp/chandelier/lamp_body': green_l,
    'lamp/ceiling_lamp/chandelier/chain': green_d,

    'lamp/ceiling_lamp/pendant_lamp/pendant_lamp_unit/lamp_head': orange_l,
    'lamp/ceiling_lamp/pendant_lamp/pendant_lamp_unit/chain': cardinal,

    'lamp/ceiling_lamp/pendant_lamp/lamp_base/lamp_holistic_base/lamp_base_part': blue_a,
    
    'lamp/table_or_floor_lamp/lamp_base/lamp_holistic_base/lamp_base_part': blue_l,
    'lamp/table_or_floor_lamp/lamp_base/lamp_leg_base/leg': blue_n,

    'lamp/table_or_floor_lamp/lamp_body/lamp_body_jointed': yellow_d,
    'lamp/table_or_floor_lamp/lamp_body/lamp_pole': lime,
    'lamp/table_or_floor_lamp/lamp_body/lamp_body_solid': gold,
    
    'lamp/table_or_floor_lamp/lamp_unit/lamp_head': mocha,
    'lamp/table_or_floor_lamp/lamp_unit/lamp_arm': dark_brown,
    'lamp/table_or_floor_lamp/lamp_unit/connector': green_d,
    
    'lamp/table_or_floor_lamp/power_cord/cord': pink,
        
    'lamp/wall_lamp/lamp_base/lamp_holistic_base/lamp_base_part': turqoise,    

    'lamp/wall_lamp/lamp_unit/lamp_head': fush,
    'lamp/wall_lamp/lamp_unit/lamp_arm': teal,
            
    'lamp/street_lamp/lamp_post': mint,

    'lamp/street_lamp/street_lamp_base': baby_blue,

    'lamp/street_lamp/lamp_unit/lamp_arm': cafe,
    'lamp/street_lamp/lamp_unit/lamp_head': teal,            
    
    # STORAGE
    
    'storage_furniture/cabinet/shelf': fush,

    'storage_furniture/cabinet/cabinet_frame/bottom_panel': red_l,
    'storage_furniture/cabinet/cabinet_frame/top_panel': red_d,
    
    'storage_furniture/cabinet/cabinet_frame/frame_horizontal_bar': mint,
    'storage_furniture/cabinet/cabinet_frame/frame_vertical_bar': turqoise,
    'storage_furniture/cabinet/cabinet_frame/vertical_side_panel': baby_blue,
    'storage_furniture/cabinet/cabinet_frame/vertical_front_panel': blue_l,
    'storage_furniture/cabinet/cabinet_frame/vertical_divider_panel': blue_n,
    'storage_furniture/cabinet/cabinet_frame/back_panel': cafe,
    
    'storage_furniture/cabinet/cabinet_base/foot_base': pink,
    
    'storage_furniture/cabinet/drawer/handle': yellow_d,
    'storage_furniture/cabinet/drawer/drawer_box/drawer_front': yellow_l, 
      
    'storage_furniture/cabinet/countertop': green_l,    
    'storage_furniture/cabinet/cabinet_door': green_d,
    
}

def do_vis(in_folder, out_name, REGION=False):
    if REGION:
        cmap = {i:data_utils.get_color(i) for i in range(100)}
        PMAP = {i:i for i in range(100)}
    else:
        cmap = CMAP
        PMAP = {}
        
    parts = []

    with open(f'{in_folder}/info.txt') as f:
        for line in f:
            ls = line.split()
            pn = ls[1]
            name = ls[5]

            if int(pn) in PMAP:
                name = PMAP[int(pn)]
            
            v,f = utils.loadObj(f'{in_folder}/part_{pn}.obj')
            color = cmap[name]
            
            parts.append((v, f, color))

    face_offset = 1
    
    o_verts = []
    o_faces = []

    for (verts, faces, color) in parts:
        _fo = 0
    
        for a, b, c in verts:
            o_verts.append(f'v {a} {b} {c} {color[0]} {color[1]} {color[2]}\n')
            _fo += 1

        for a, b, c in faces:
            o_faces.append(f'f {a+face_offset} {b+face_offset} {c+face_offset}\n')
            
        face_offset += _fo

    with open(out_name, 'w') as f:
        for v in o_verts:
            f.write(v)
            
        for fa in o_faces:
            f.write(fa)


def get_parts(in_folder):
    parts = set()

    with open(f'{in_folder}/info.txt') as f:
        for line in f:
            ls = line.split()
            pn = ls[1]
            name = ls[5]

            v,f = utils.loadObj(f'{in_folder}/part_{pn}.obj')

            parts.add(name)
                
    return parts

# PartNet IDS of qualitative examples in paper + supplemental
EXAMPLES = [
    ('chair', 2913),
    ('chair', 36908),
    ('chair', 40592),
    ('chair', 43126),

    ('table', 20828),
    ('table', 21909),
    ('table', 27441),
    ('table', 28880),
    ('table', 32767),

    ('lamp', 14177),
    ('lamp', 14271),
    ('lamp', 17083),
    ('lamp', 14064),

    ('vase', 4160),
    ('vase', 4302),
    ('vase', 6374),

    ('knife', 433),
    ('knife', 1147),
    ('knife', 1217),
    
    ('storagefurniture', 46048),
    ('storagefurniture', 48494),
    ('storagefurniture', 49027),
]




    
