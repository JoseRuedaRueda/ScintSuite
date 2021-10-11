"""
Side auxiliar functions

Jose Rueda: jrrueda@us.es

Introduced in version 0.6.0
"""


# -----------------------------------------------------------------------------
# --- Dictionaries
# -----------------------------------------------------------------------------
def update_case_insensitive(a, b):
    """
    Update a dictionary avoiding problems due to case sensitivity

    Jose Rueda Rueda: jrrueda@us.es

    Note: This is a non-perfectly efficient workaround. Please do not use it
    routinely inside heavy loops. It will only change in a the fields contained
    in b, it will not create new fields in a

    Please, Pablo, do not kill me for this extremelly uneficient way of doing
    this

    @params a: Main dictionary
    @params b: Dictionary with the extra information to include in a
    """

    keys_a_lower = [key.lower() for key in a.keys()]
    keys_a = [key for key in a.keys()]
    keys_b_lower = [key.lower() for key in b.keys()]
    keys_b = [key for key in b.keys()]

    for k in keys_b_lower:
        if k in keys_a_lower:
            for i in range(len(keys_a_lower)):
                if keys_a_lower[i] == k:
                    keya = keys_a[i]
            for i in range(len(keys_b_lower)):
                if keys_b_lower[i] == k:
                    keyb = keys_b[i]
            a[keya] = b[keyb]
