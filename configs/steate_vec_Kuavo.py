STATE_VEC_IDX_MAPPING = {
    # 6D representation of left end effector (EEF) (indices 0–5)
    'left_eef_x': 0,
    'left_eef_y': 1,
    'left_eef_z': 2,
    'left_eef_rx': 3,
    'left_eef_ry': 4,
    'left_eef_rz': 5,
    # Orientation of left hand in RPY (indices 6–8)
    'left_roll': 6,
    'left_pitch': 7,
    'left_yaw': 8,
    # Left hand gripper (index 9)
    'left_gripper': 9,
    # 6D representation of right end effector (EEF) (indices 10–15)
    'right_eef_x': 10,
    'right_eef_y': 11,
    'right_eef_z': 12,
    'right_eef_rx': 13,
    'right_eef_ry': 14,
    'right_eef_rz': 15,
    # Orientation of right hand in RPY (indices 16–18)
    'right_roll': 16,
    'right_pitch': 17,
    'right_yaw': 18,
    # Right hand gripper (index 19)
    'right_gripper': 19,
}

STATE_VEC_LEN = len(STATE_VEC_IDX_MAPPING)  # = 20
