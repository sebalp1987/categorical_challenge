
def replace_dict(x, dict_values):

    for key, value in dict_values.items():
        x = x.replace(key, value)

    return x