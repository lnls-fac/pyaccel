
interactive_list = []

def interactive(obj):
    interactive_list.append(
        {
            'module': obj.__module__,
            'name': obj.__name__
        }
    )

    return obj
