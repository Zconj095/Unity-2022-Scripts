def calculate_complexity(C_move, C_time, C_space, w1, w2, w3):
    return w1 * C_move + w2 * C_time + w3 * C_space

def calculate_skill_level(S_exp, S_rate, RS):
    return S_exp * S_rate + RS

import numpy as np

def reaction_effect(RT, RT_threshold):
    return 1 / (1 + np.exp(-(RT - RT_threshold)))

def response_speed_effect(RS_base, RS_factor, RS):
    return RS_base * (1 + RS_factor * RS)

def calculate_oss_effectiveness(Base_effect, C_move, C_time, C_space, w1, w2, w3, S_exp, S_rate, RS, RT, RT_threshold, RS_base, RS_factor, U, E, Env_effect, Eq_synergy, Resource_effect, Opp_effect):
    C = calculate_complexity(C_move, C_time, C_space, w1, w2, w3)
    S = calculate_skill_level(S_exp, S_rate, RS)
    Effect_RT = reaction_effect(RT, RT_threshold)
    Speed_RS = response_speed_effect(RS_base, RS_factor, RS)
    return ((Base_effect + C + S + Effect_RT) * U * Speed_RS - E) * Env_effect * Eq_synergy * Resource_effect * Opp_effect

