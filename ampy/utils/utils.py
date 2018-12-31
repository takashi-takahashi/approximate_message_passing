# coding=utf-8


def update_dumping(old_x, new_x, dumping_coefficient):
    return dumping_coefficient * new_x + (1.0 - dumping_coefficient) * old_x
