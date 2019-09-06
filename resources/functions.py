
def replace_dict(x, dict_values):

    for key, value in dict_values.items():
        value = str(value)
        x = x.replace(key, value)

    return x